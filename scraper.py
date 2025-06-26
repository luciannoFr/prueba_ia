import requests
from bs4 import BeautifulSoup
import re
import logging
from urllib.parse import quote
import os 

from config import BASE_URL, KNOWLEDGE_BASE_FILE
from data_manager import load_knowledge_base, save_knowledge_base 

logger = logging.getLogger(__name__)

def scrape_tramite_data(url):
    """
    Performs web scraping of the procedure page on formosa.gob.ar and extract all information.
    Adapts extraction to search for content within tabs such as "Detalle", "Cuánto sale",
    "Dónde se realiza", "Formularios" and "Normas".
    Includes basic caching logic.
    """
    knowledge_base = load_knowledge_base()
    
    for entry in knowledge_base:
        if entry.get('url') == url and entry.get('data'):
            logger.info(f"Returning cached data for {url}")
            return entry.get('data') 

    logger.info(f"Scraping new data for {url}")
    datos_tramite = {
        'titulo': '',
        'descripcion': '',
        'requisitos': [],
        'observaciones': '',
        'pasos': [],
        'costo': '',
        'direccion': '',
        'coordenadas': '',
        'horarios': '',
        'telefono': '',
        'email': '',
        'sitio': '',
        'responsable': '',
        'modalidad': '',
        'mapa_url': '',
        'tiene_opciones_ubicacion': False,
        'opciones_ubicacion': [],
        'necesita_seleccion': False,
        'formularios': []
    }

    try:
        response = requests.get(url, timeout=15) 
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')

        datos_tramite['titulo'] = soup.find('h2').get_text(strip=True) if soup.find('h2') else "Trámite no especificado"

        content_div = soup.find('div', id='content')
        if content_div:
            tabs_nav = content_div.find('ul', class_='nav-tabs')
            if tabs_nav:
                for tab_link in tabs_nav.find_all('a'):
                    tab_name = tab_link.get_text(strip=True).lower()
                    tab_id = tab_link.get('href', '').replace('#', '')
                    tab_pane = content_div.find('div', id=tab_id)
                    if not tab_pane:
                        continue

                    if "detalle" in tab_name:
                        desc_callout = tab_pane.find('div', class_='bs-callout bs-callout-info')
                        if desc_callout:
                            datos_tramite['descripcion'] = desc_callout.find('p').get_text(strip=True) if desc_callout.find('p') else ""
                        req_callout = tab_pane.find('div', class_='bs-callout bs-callout-warning')
                        if req_callout:
                            req_p = req_callout.find('p')
                            if req_p:
                                req_text = req_p.get_text(separator='\n').strip()
                                requisitos = [line.strip() for line in req_text.split('\n') if line.strip() and len(line.strip()) > 5]
                                if requisitos:
                                    datos_tramite['requisitos'] = requisitos
                                else:
                                    datos_tramite['requisitos'] = ["No se encontraron requisitos específicos."]

                    elif "observaciones" in tab_name or "observacion" in tab_name:
                        obs_callout = tab_pane.find('div', class_='bs-callout bs-callout-danger')
                        if obs_callout:
                            obs_text = obs_callout.get_text(strip=True)
                            if obs_text:
                                obs_lines = re.split(r'<br\s*/>|\n|\.,\s*', obs_text)
                                observaciones = [line.strip() for line in obs_lines if line.strip() and len(line.strip()) > 5]
                                if len(observaciones) == 1 and len(observaciones[0]) > len(obs_text) * 0.8:
                                    observaciones = obs_text # Keep as single string if it's one long paragraph
                                datos_tramite['observaciones'] = observaciones
                        else:
                            datos_tramite['observaciones'] = "No se encontraron observaciones."

                    elif "formularios" in tab_name:
                        form_links = tab_pane.find_all('a', href=True)
                        for link in form_links:
                            href = link['href'].lower()
                            if href.endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx')):
                                form_name = link.get_text(strip=True)
                                if not form_name or len(form_name) < 3: # Try to get text from previous sibling if link text is too short
                                    prev = link.find_previous(string=True)
                                    form_name = prev.strip() if prev and len(prev.strip()) > 2 else "Formulario sin nombre"
                                form_url = link['href']
                                if not form_url.startswith('http'):
                                    if form_url.startswith('/'):
                                        form_url = BASE_URL + form_url
                                    else:
                                        base_url_current = '/'.join(url.split('/')[:3])
                                        form_url = os.path.join(base_url_current, form_url.lstrip('/'))
                                datos_tramite['formularios'].append({'nombre': form_name, 'url': form_url})

                    elif "cuánto sale" in tab_name or "cuanto" in tab_name:
                        cost_table = tab_pane.find('table')
                        if cost_table:
                            costos = []
                            for row in cost_table.find_all('tr'):
                                cols = row.find_all('td')
                                if len(cols) >= 2:
                                    desc = cols[0].get_text(strip=True)
                                    valor = cols[1].get_text(strip=True)
                                    costos.append({'descripcion': desc, 'valor': valor})
                            datos_tramite['costo'] = costos if costos else "No se especifica costo."
                        else:
                            cost_text_match = re.search(r'Costo:\s*(.+)|Precio:\s*(.+)|(\$[\d\.,]+(?: ARS)?)', tab_pane.get_text(), re.IGNORECASE)
                            if cost_text_match:
                                datos_tramite['costo'] = cost_text_match.group(1) or cost_text_match.group(2) or cost_text_match.group(3)

                    elif "dónde se realiza" in tab_name or "donde" in tab_name:
                        panel_groups = tab_pane.find('div', class_='panel-group')
                        if panel_groups:
                            locations_found = []
                            for panel in panel_groups.find_all('div', class_='panel-default'):
                                loc_name_tag = panel.find('div', class_='panel-title')
                                if loc_name_tag:
                                    loc_name = loc_name_tag.get_text(strip=True).replace('Dirección del ', '')
                                    loc_details = {}
                                    panel_body = panel.find('div', class_='panel-body')
                                    if panel_body:
                                        for row in panel_body.find_all('tr'):
                                            cols = row.find_all('td')
                                            if len(cols) == 2:
                                                key = cols[0].get_text(strip=True).replace(':', '').lower()
                                                value = cols[1].get_text(strip=True)
                                                if key == "domicilio":
                                                    loc_details['direccion'] = value
                                                elif key == "teléfono":
                                                    loc_details['telefono'] = value
                                                elif key == "e-mail":
                                                    loc_details['email'] = value
                                                elif key == "horario de atención":
                                                    loc_details['horarios'] = value
                                                elif key == "responsable":
                                                    loc_details['responsable'] = value
                                    if loc_name and loc_details:
                                        locations_found.append({'nombre': loc_name, **loc_details})
                            if len(locations_found) > 1:
                                datos_tramite['tiene_opciones_ubicacion'] = True
                                datos_tramite['necesita_seleccion'] = True
                                datos_tramite['opciones_ubicacion'] = locations_found
                            elif len(locations_found) == 1:
                                datos_tramite['direccion'] = locations_found[0].get('direccion', '')
                                datos_tramite['telefono'] = locations_found[0].get('telefono', '')
                                datos_tramite['email'] = locations_found[0].get('email', '')
                                datos_tramite['horarios'] = locations_found[0].get('horarios', '')
                                datos_tramite['responsable'] = locations_found[0].get('responsable', '')

                        if not datos_tramite['direccion'] and not datos_tramite['opciones_ubicacion']:
                            address_patterns = [
                                re.compile(r'(Sarmiento|Av\.|Avenida|Calle|Jonas Salk|25 de Mayo|Saavedra|Rivadavia|Moreno|Belgrano|Roca|Salta|Mitre)\s*(Nº)?\s*\d+\s*(?:[a-zA-ZáéíóúÁÉÍÓÚñÑ\s\d\-"\']*)?(?:CP\s*\d{4})?', re.IGNORECASE),
                                re.compile(r'Domicilio\s*:\s*(.+)', re.IGNORECASE),
                                re.compile(r'Direcci[óo]n\s*:\s*(.+)', re.IGNORECASE)
                            ]
                            for pattern in address_patterns:
                                match = pattern.search(tab_pane.get_text())
                                if match:
                                    datos_tramite['direccion'] = match.group(0).replace('Domicilio:', '').replace('Dirección:', '').strip()
                                    tel_match = re.search(r'Teléfono:\s*(\(?\d{3,4}\)?[\s-]?\d{6,8})', tab_pane.get_text())
                                    if tel_match: datos_tramite['telefono'] = tel_match.group(1).strip()
                                    email_match = re.search(r'E-mail:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', tab_pane.get_text())
                                    if email_match: datos_tramite['email'] = email_match.group(1).strip()
                                    horarios_match = re.search(r'Horario de Atención:\s*(.+)', tab_pane.get_text())
                                    if horarios_match: datos_tramite['horarios'] = horarios_match.group(1).strip()
                                    break

        modalidad_block = soup.find(lambda tag: tag.name in ['h3', 'h4', 'h5', 'p'] and 'Cómo se realiza el trámite?' in tag.get_text(strip=True))
        if modalidad_block:
            modalidad_text_tag = modalidad_block.find_next('p')
            if modalidad_text_tag:
                datos_tramite['modalidad'] = modalidad_text_tag.get_text(strip=True)

        sitio_block = soup.find(lambda tag: tag.name in ['h3', 'h4', 'h5', 'p'] and 'Sitio Oficial:' in tag.get_text(strip=True))
        if sitio_block:
            sitio_link = sitio_block.find('a', href=True)
            if sitio_link:
                datos_tramite['sitio'] = sitio_link.get('href')

        if datos_tramite['direccion']:
            direccion_completa = f"{datos_tramite['direccion']}, Formosa" if "formosa" not in datos_tramite['direccion'].lower() else datos_tramite['direccion']
            encoded_address = quote(direccion_completa)
            datos_tramite['mapa_url'] = f"https://www.google.com/maps/search/?api=1&query={encoded_address}"
        if "verificacion-fisica-del-automotor" in url.lower() and not datos_tramite['direccion']:
            datos_tramite['direccion'] = "Unidad de Tránsito - Pringles y Rivadavia, Formosa Capital"
            datos_tramite['mapa_url'] = f"https://www.google.com/maps/search/?api=1&query={quote('Pringles y Rivadavia, Formosa Capital')}"

        os.makedirs(os.path.dirname(KNOWLEDGE_BASE_FILE), exist_ok=True)
        found_in_kb = False
        for i, entry in enumerate(knowledge_base):
            if entry.get('url') == url:
                knowledge_base[i]['data'] = datos_tramite
                found_in_kb = True
                break
        if not found_in_kb:
            knowledge_base.append({"url": url, "data": datos_tramite})
            
        save_knowledge_base(knowledge_base)
        return datos_tramite

    except requests.exceptions.RequestException as e:
        logger.error(f"Error de red o HTTP al hacer scraping en {url}: {e}")
        return None