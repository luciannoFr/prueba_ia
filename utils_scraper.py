import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
import re
import os
import json
import time

BASE_URL = "https://www.formosa.gob.ar"
PAGINAS_INICIALES = [
    f"{BASE_URL}/tramites/organismos",
    f"{BASE_URL}/tramites/temas",
    f"{BASE_URL}/tramites/destinatarios",
    f"{BASE_URL}/tramites/buscar"  # Página de búsqueda principal
]
OUTPUT_DIR = "data"
TRAMITES_URLS_FILE = os.path.join(OUTPUT_DIR, "tramites_urls.json")

PATRON_TRAMITE = re.compile(r"/tramite/\d+/[\w\-áéíóúÁÉÍÓÚñÑ]+")
PATRON_PAGINACION = re.compile(r"/tramites/buscar/pagina/\d+")

session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retries))

def descubrir_urls_tramites():
    """
    Recorre todas las páginas de trámites de forma exhaustiva,
    explorando enlaces internos y extrayendo URLs únicas de trámite.
    """
    urls_tramites = set()
    urls_visitadas = set()
    urls_por_visitar = set(PAGINAS_INICIALES)

    for pagina in range(1, 25):  # Del 1 al 24
        url_paginacion = f"{BASE_URL}/tramites/buscar/pagina/{pagina}"
        urls_por_visitar.add(url_paginacion)

    while urls_por_visitar:
        url_actual = urls_por_visitar.pop()
        url_actual = urldefrag(url_actual).url
        if url_actual in urls_visitadas:
            continue

        print(f"Visitando: {url_actual}")
        try:
            resp = session.get(url_actual, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Falló GET {url_actual}: {e}")
            continue

        urls_visitadas.add(url_actual)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Buscar enlaces de trámites
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            href = urljoin(BASE_URL, href)
            href, _ = urldefrag(href)
            if not href.startswith(BASE_URL):
                continue
            if PATRON_TRAMITE.search(href):
                urls_tramites.add(href)
            elif PATRON_PAGINACION.search(href) or "/tramites" in href:
                if href not in urls_visitadas and href not in urls_por_visitar:
                    urls_por_visitar.add(href)

        for li in soup.select("ul.pagination li a"):
            href = li.get("href")
            if href:
                href = urljoin(BASE_URL, href)
                href, _ = urldefrag(href)
                if PATRON_PAGINACION.search(href) and href not in urls_visitadas and href not in urls_por_visitar:
                    urls_por_visitar.add(href)

        time.sleep(1) 

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TRAMITES_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(urls_tramites), f, ensure_ascii=False, indent=2)

    print(f"Se descubrieron {len(urls_tramites)} URLs únicas de trámites.")
    return urls_tramites

def procesar_todos_los_tramites():
    """
    Procesa cada trámite llamando a scrape_tramite_data().
    """
    if not os.path.exists(TRAMITES_URLS_FILE):
        print("Primero ejecutá descubrir_urls_tramites()")
        return

    with open(TRAMITES_URLS_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    print(f"🔎 Procesando {len(urls)} trámites...")
    from scraper import scrape_tramite_data

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Scrapeando: {url}")
        datos = scrape_tramite_data(url)
        if datos:
            print(f"[OK] {datos.get('titulo', 'sin título')}")
        else:
            print(f"[ERROR] Falló en {url}")

if __name__ == "__main__":
    descubrir_urls_tramites()
    procesar_todos_los_tramites()