import os
import nbformat
from nbconvert import HTMLExporter

src_path = '../'
filenames = [os.path.join(src_path, f) for f in os.listdir(src_path)
             if f.lower().endswith('.ipynb')]

for fp in filenames:

    # read jupyter notebook file
    jake_notebook = nbformat.read(fp, as_version=4)

    # instantiate the exporter
    html_exporter = HTMLExporter()
    html_exporter.template_name = 'classic'     # 'all', 'basic'

    # convert the notebook
    body, resources = html_exporter.from_notebook_node(jake_notebook)

    # fix relative image links by accounting for child folder location
    body = body.replace('./img/', '../img/')

    # save content as html file
    fname = os.path.splitext(os.path.basename(fp))[0]+".html"
    dir_path = os.path.join(os.path.dirname(os.path.abspath(fp)), 'nbconv_html')
    os.mkdir(dir_path) if not os.path.exists(dir_path) else None
    with open(os.path.join(dir_path, fname), "w") as file:
        file.write(body)
