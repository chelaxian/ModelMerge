import json

from ..src.ModelMerge.plugins.websearch import get_search_results
from ..src.ModelMerge.plugins.arXiv import download_read_arxiv_pdf
from ..src.ModelMerge.plugins.image import generate_image
from ..src.ModelMerge.plugins.today import get_date_time_weekday
from ..src.ModelMerge.plugins.run_python import run_python_script

from ..src.ModelMerge.plugins.config import function_to_json


print(json.dumps(function_to_json(get_search_results), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(download_read_arxiv_pdf), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(generate_image), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(get_date_time_weekday), indent=4, ensure_ascii=False))
print(json.dumps(function_to_json(run_python_script), indent=4, ensure_ascii=False))
