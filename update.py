import os
import datetime
import pytz
import yaml
import re

DEFAULT_CONTENT = """[WIP]\n\n"""

UPDATE_TIME_START_COMMENT = "<!-- update-time-start -->"
UPDATE_TIME_END_COMMENT = "<!-- update-time-end -->"
MAIN_START_COMMENT = "<!-- main-start -->"
MAIN_END_COMMENT = "<!-- main-end -->"


def get_markdown_url(
    title: str, abbr: str = None, conf: str = None, links: dict = None
) -> str:
    """Get markdown url"""
    links_str = ""
    if links is not None:
        for link_item in links:
            links_str += f" [[{link_item}]]({links[link_item]})"
    abbr_str = "" if abbr is None or abbr == "" else f"**[{abbr}]** "
    conf_str = "" if conf is None else f"(_{conf}_)"
    return f"{abbr_str}{title} {conf_str}{links_str}"


def generate_new_readme(
    src: str, content: str, start_comment: str, end_comment: str
) -> str:
    """Generate a new Readme.md"""
    pattern = f"{start_comment}[\\s\\S]+{end_comment}"
    repl = f"{start_comment}\n\n{content}\n\n{end_comment}"
    if re.search(pattern, src) is None:
        print(
            f"can not find section in src, please check it, it should be {start_comment} and {end_comment}"
        )
    return re.sub(pattern, repl, src)


def write_content(content: str, start_comment: str, end_comment: str):
    with open("README.md", "r", encoding="utf-8") as f:
        src = f.read()
    new_src = generate_new_readme(src, content, start_comment, end_comment)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_src)


def generate_main():
    papers = yaml.safe_load(open("data.yaml", encoding="utf-8"))
    last_category = ''
    content = ""
    paper_set = set()
    for topic in papers:
        topic_name = topic
        
        topic = papers[topic]
        
        category = topic.get('category')
        if category != last_category:
            content += f'## {category}\n'
            last_category = category
        content += f'### {topic_name}\n'
            
        
        
        list_items = topic.get('list', [])
        paper_meta = []
        
        for item in list_items:
            title = item.get('title', '')
            abbr = item.get('abbr', 'N/A')
            venue = item.get('venue', '')
            year = item.get('year', '')
            links = item.get('links', None)
            if links is not None:
                links = {k: v for k, v in links.items() if v is not None}
            paper_meta.append(
                {
                    "title": title,
                    "abbr": abbr,
                    "year": year,
                    "venue": venue,
                    "links": links,
                }
            )
            paper_set.add(title)
        sorted_paper_meta = sorted(paper_meta, key=lambda x: x['year'], reverse=True)
        markdown_table = "| Title | Abbreviation | Venue | Year | Materials |\n"
        markdown_table += "|----------|----|----|----|----|\n"
        
        for item in sorted_paper_meta:
            title = item.get('title')
            abbr = item.get('abbr')
            venue = item.get('venue')
            year = item.get('year')
            
            markdown_table += f"| {title} | {abbr} | {venue} | {year} | "
            links = item.get('links', None)
            if links is not None:
                for k, v in links.items():
                    markdown_table += f"[[{k}]({v})] "
            markdown_table += "|\n"

        content += markdown_table + "\n"

    print(content)
    print(len(paper_set))
    content += f'{len(paper_set)} papers are included'
    write_content(content, MAIN_START_COMMENT, MAIN_END_COMMENT)


def update_time():
    tz = pytz.timezone("Asia/Shanghai")
    update_time_str = datetime.datetime.now(tz).strftime("%b %d, %Y %H:%M:%S")
    update_time_str = f"**Last Update: {update_time_str}**"
    print(update_time_str)
    write_content(update_time_str, UPDATE_TIME_START_COMMENT, UPDATE_TIME_END_COMMENT)


if __name__ == "__main__":
    generate_main()
    update_time()
