import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urldefrag, quote
from bs4 import BeautifulSoup
import re
import os
import json
import logging

from config import BASE_URL, KNOWLEDGE_BASE_FILE
from data_manager import load_knowledge_base, save_knowledge_base

logger = logging.getLogger(__name__)

session = requests.Session()
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

PHONE_REGEX = re.compile(r"\+?54?\s*\(?0?370\)?[\s-]?\d+")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def normalize_phone(raw):
    digits = re.sub(r"\D", "", raw)
    if digits.startswith('0'):
        digits = digits.lstrip('0')
    return f"+54 {digits}"


def normalize_cost(raw):
    match = re.search(r"\d+[\.,]?\d*", raw.replace('ARS',''))
    return f"${match.group(0)}" if match else raw.strip()


def split_address(raw):
    return {'full': raw.strip()}


def scrape_tramite_data(url):
    kb = load_knowledge_base()
    for entry in kb:
        if entry.get('url') == url and entry.get('data'):
            logger.info(f"Cached: {url}")
            return entry['data']

    logger.info(f"Scraping: {url}")
    data = {
        'titulo': None,
        'descripcion': None,
        'requisitos': [],
        'costo': None,
        'direccion': None,
        'horarios': None,
        'telefono': None,
        'email': None,
        'formularios': [],
        'observaciones': [],
        'pasos': [],
        'modalidad': None,
        'duracion': None,
        'normativa': [],
        'destinatario': None,
        'categoria': None,
        'organismo': None,
        'sitio_oficial': None,
        'similares': [],
        'externos': [],
        'coordenadas': None,
        'mapa_url': None,
        'opciones_ubicacion': []
    }

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        # Título y descripción
        data['titulo'] = soup.find('h2').get_text(strip=True) if soup.find('h2') else None
        desc_p = soup.select_one('.bs-callout-info p')
        data['descripcion'] = desc_p.get_text(' ', strip=True) if desc_p else None

        # Requisitos y observaciones
        data['requisitos'] = [p.get_text(strip=True) for p in soup.select('.bs-callout-warning p')]
        data['observaciones'] = [p.get_text(strip=True) for p in soup.select('.bs-callout-danger p')]

        # Formularios
        name = None
        for row in soup.select('#formularios table tr'):
            strong = row.find('strong')
            if strong:
                name = strong.get_text(strip=True)
            link = row.find('a', href=True)
            if link and name:
                full_url = urljoin(BASE_URL, link['href'])
                data['formularios'].append({'nombre': name, 'url': full_url})
                name = None

        # Normas
        name_n = None
        for row in soup.select('#normas table tr'):
            strong = row.find('strong')
            if strong:
                name_n = strong.get_text(strip=True)
            link = row.find('a', href=True)
            if link and name_n:
                full_url = urljoin(BASE_URL, link['href'])
                data['normativa'].append({'nombre': name_n, 'url': full_url})
                name_n = None

        # Costo
        cost_table = soup.select_one('#cuanto table')
        if cost_table:
            costos = []
            for row in cost_table.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 2:
                    descripcion = cols[0].get_text(" ", strip=True)
                    valor = cols[1].get_text(" ", strip=True)
                    costos.append({
                        'descripcion': descripcion,
                        'valor': valor
                    })
            data['costo'] = costos
        else:
            raw_cost = soup.find(text=re.compile(r'\$\s*\d+'))
            data['costo'] = [{'descripcion': None, 'valor': raw_cost.strip()}] if raw_cost else []
            
        # Ubicaciones: extraer múltiples sedes con detalle
        for panel in soup.select('#donde .panel-default'):
            title_tag = panel.select_one('.panel-title a')
            body = panel.select_one('.panel-body')
            if not (title_tag and body):
                continue
            loc = {'nombre': title_tag.get_text(strip=True)}
            # Recorrer filas de la tabla dentro del panel
            for tr in body.select('tr'):
                cols = tr.find_all('td')
                if len(cols) != 2:
                    continue
                key = cols[0].get_text(strip=True).rstrip(':').lower()
                val = cols[1].get_text(' ', strip=True)
                if 'domicilio' in key:
                    loc['direccion'] = val
                elif key.startswith('tel'):
                    loc.setdefault('telefonos', []).append(normalize_phone(val))
                elif 'e-mail' in key or 'email' in key:
                    loc['email'] = val
                elif 'responsable' in key:
                    loc['responsable'] = val
                elif 'horario' in key:
                    loc['horarios'] = val
            data['opciones_ubicacion'].append(loc)

        if len(data['opciones_ubicacion']) == 1:
            loc0 = data['opciones_ubicacion'][0]
            data['direccion'] = loc0.get('direccion')
            data['telefono'] = loc0.get('telefonos')
            data['email'] = loc0.get('email')
            data['horarios'] = loc0.get('horarios')

        # Pasos detallados
        for step in soup.select('.steps .step'):
            num = step.select_one('.number')
            title_s = step.select_one('.step-wrapper h4')
            desc_p = step.select_one('.step-wrapper p')
            data['pasos'].append({
                'numero': num.get_text(strip=True) if num else None,
                'titulo': title_s.get_text(strip=True) if title_s else None,
                'descripcion': desc_p.get_text(strip=True) if desc_p else None
            })

        # Bloques de features adicionales (destinatario, categoría, etc.)
        for fb in soup.select('.text-small.features-block'):
            txt = fb.get_text(' ', strip=True)
            if 'Trámite destinado a' in txt:
                a = fb.find('a', href=True)
                data['destinatario'] = a.get_text(strip=True) if a else None
            if 'Tema:' in txt:
                data['categoria'] = txt.split('Tema:',1)[1].strip()
            if 'Organismo Responsable' in txt:
                a = fb.find('a', href=True)
                data['organismo'] = a.get_text(strip=True) if a else None
            if 'Sitio Oficial' in txt:
                a = fb.find('a', href=True)
                data['sitio_oficial'] = urljoin(BASE_URL, a['href']) if a else None
            if 'Duración Aproximada' in txt:
                h6 = fb.find('h6')
                data['duracion'] = h6.get_text(strip=True) if h6 else None
            if 'Cómo se realiza' in txt:
                strong = fb.find('strong')
                data['modalidad'] = strong.get_text(strip=True) if strong else data['modalidad']
            if 'Trámites similares' in txt:
                for a in fb.select('.list-group a[href]'):
                    data['similares'].append({'nombre': a.get_text(strip=True), 'url': urljoin(BASE_URL, a['href'])})
            if 'Trámites Externos' in txt:
                for a in fb.select('.list-group a[href]'):
                    data['externos'].append({'nombre': a.get_text(strip=True), 'url': a['href']})

        # Mapa usando dirección simple
        if data.get('direccion'):
            addr_text = data['direccion'].get('full', '') if isinstance(data['direccion'], dict) else ''
            q = quote(addr_text + ', Formosa')
            data['mapa_url'] = f"https://www.google.com/maps/search/?api=1&query={q}"

        os.makedirs(os.path.dirname(KNOWLEDGE_BASE_FILE), exist_ok=True)
        found = False
        for i, e in enumerate(kb):
            if e['url'] == url:
                kb[i]['data'] = data
                found = True
                break
        if not found:
            kb.append({'url': url, 'data': data})
        save_knowledge_base(kb)
        return data
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None

if __name__ == '__main__':
    from utils_scraper import descubrir_urls_tramites, procesar_todos_los_tramites
    urls = descubrir_urls_tramites()
    procesar_todos_los_tramites()
