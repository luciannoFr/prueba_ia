import requests
from bs4 import BeautifulSoup
import re
import os
import json
from scraper import scrape_tramite_data

BASE_URL = "https://www.formosa.gob.ar"
PAGINAS_LISTADO = [
    f"{BASE_URL}/tramites",
    f"{BASE_URL}/tramites/organismos",
    f"{BASE_URL}/tramites/temas",
    f"{BASE_URL}/tramites/destinatarios"
]

TRAMITES_URLS_FILE = "data/tramites_urls.json"


def descubrir_urls_tramites():
    """
    Recorrido más profundo: navega por subpáginas y encuentra todas las URLs de trámites.
    """
    urls_tramites = set()
    urls_visitadas = set()
    urls_por_visitar = set(PAGINAS_LISTADO)
    patron_tramite = re.compile(r"/tramite/\d+/[a-zA-Z0-9_\-]+")

    while urls_por_visitar:
        url_actual = urls_por_visitar.pop()
        if url_actual in urls_visitadas:
            continue
        print(f"[WEB] Visitando: {url_actual}")
        try:
            resp = requests.get(url_actual, timeout=50)
            soup = BeautifulSoup(resp.text, "html.parser")
            urls_visitadas.add(url_actual)

            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href.startswith("http"):
                    href = BASE_URL + href if href.startswith("/") else f"{BASE_URL}/{href}"
                
                if "/tramite/" in href and patron_tramite.search(href):
                    urls_tramites.add(href)
                elif "/tramites/" in href or "/tramite/" in href:
                    if href not in urls_visitadas:
                        urls_por_visitar.add(href)
        except Exception as e:
            print(f"[ERROR] Error en {url_actual}: {e}")

    os.makedirs("data", exist_ok=True)
    with open(TRAMITES_URLS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(urls_tramites), f, indent=2, ensure_ascii=False)

    print(f"/*\ Se descubrieron {len(urls_tramites)} URLs únicas de trámites.")



def procesar_todos_los_tramites():
    """
    Carga las URLs descubiertas y las procesa con scrape_tramite_data()
    """
    if not os.path.exists(TRAMITES_URLS_FILE):
        print("Primero ejecutá descubrir_urls_tramites()")
        return

    with open(TRAMITES_URLS_FILE, "r", encoding="utf-8") as f:
        urls = json.load(f)

    print(f"🔎 Procesando {len(urls)} trámites...")
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Scrapeando: {url}")
        datos = scrape_tramite_data(url)
        if datos:
            print(f"[OK] {datos['titulo']}")
        else:
            print(f"[ERROR] Error en {url}")


if __name__ == "__main__":
    descubrir_urls_tramites()
    procesar_todos_los_tramites()
