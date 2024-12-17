"""
Scrape the arXiv category taxonomy to create `data/dataset/categories.csv`.
"""
import re

import pandas as pd
from requests_cache import CachedSession
from lxml import etree


def main():
    session = CachedSession()
    r = session.get("https://arxiv.org/category_taxonomy")
    r.raise_for_status()
    html = etree.HTML(r.content)
    taxonomy = html.cssselect("#category_taxonomy_list")[0]

    categories = []
    for div in taxonomy.cssselect("div.accordion-body > .columns .divided"):
        columns = div.cssselect("div.column")
        assert len(columns) == 2

        category = {"code": columns[0].cssselect("h4")[0].text.strip()}

        name = columns[0].cssselect("h4 > span")[0].text.strip()
        category["name"] = re.sub(r"^\(|\)$", "", name)

        category["description"] = ''.join(columns[1].itertext())

        categories.append(category)

    df = pd.DataFrame(categories)
    df.to_csv("categories.csv", index=False)


if __name__ == '__main__':
    main()