# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'DenMune'
copyright = '2021, Mohamed Abbas'
author = 'Mohamed Abbas'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'rst2pdf.pdfbuilder', 
]

# Grouping the document tree into PDF files. List of tuples
# (source start file, target name, title, author, options).
pdf_documents = [
    ('index', 'MyProject', 'My Project', 'Author Name'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'


#html_theme = "press"

#import sphinx_theme
#html_theme = "stanford_theme"
#html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
# on_rtd is whether we are on readthedocs.org, this line of code grabbed from docs.readthedocs.org
#on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
#if not on_rtd:  # only import and set the theme if we're building docs locally
#    import sphinx_theme
#    html_theme = 'stanford_theme'
#    html_theme_path = [sphinx_theme.get_html_theme_path('stanford_theme')]
# otherwise, readthedocs.org uses their theme by default, so no need to specify it


#import sphinx_bootstrap_theme
#html_theme = 'bootstrap'
#html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

#import guzzle_sphinx_theme
#html_theme_path = guzzle_sphinx_theme.html_theme_path()
#html_theme = 'guzzle_sphinx_theme'

#html_theme = "insegel"
#html_theme = "furo"

#html_theme = "sphinx_documatt_theme"
#html_theme = "pydata_sphinx_theme"
#html_theme = 'sphinx_material'

# -- Options for EPUB output
epub_show_urls = 'footnote'
