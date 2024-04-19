from bs4 import BeautifulSoup
import requests

def fetch_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Assure que la requête a réussi
        soup = BeautifulSoup(response.text, 'html.parser')
        content = soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"Erreur lors de la récupération du contenu de l'article: {e}")
        content = None
    return content

def fetch_news(api_key, query, filename):
    page = 1
    max_pages = 5  # Vous pouvez ajuster cela pour parcourir plus ou moins de pages
    url_base = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&pageSize=100'
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            while True:
                url = f"{url_base}&page={page}"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                articles = data['articles']
                if not articles:
                    break  # Si aucune article n'est retourné, arrête la boucle
                for article in articles:
                    print("Récupération du contenu pour :", article['title'])
                    content = fetch_content(article['url'])
                    if content:
                        file.write(content.replace('\n', ' ') + '\n\n')  # Écrit chaque article sur une nouvelle ligne, en remplaçant les sauts de ligne internes
                    else:
                        file.write('\n\n')  # Laisse une ligne vide si aucun contenu n'a été trouvé
                page += 1
                if page > max_pages:  # Limite le nombre de pages à parcourir
                    break
    except Exception as e:
        print(f"Erreur lors de la récupération des nouvelles : {e}")

# Exemple d'utilisation :
API_KEY = '5e62cc505c1f478abc68fc83aa6fbd30'
fetch_news(API_KEY, 'South Korea', 'news_articles.txt')
