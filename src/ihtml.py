"""
Functions to onvert dataframes to HTML and use in ipywidgets.

Requirements:
- Install for using it in jupyter:
    conda install -c conda-forge ipympl widgetsnbextension
- Or with pip
    pip install ipympl, widgetsnbextension
"""
from IPython.display import Markdown, display
from ipywidgets import *
import pandas as pd


table_style = """<style type="text/css">
    .tg  {border-collapse:collapse;border-spacing:0;border-color:#aaa;}
    .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#aaa;color:#333;background-color:#fff;}
    .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-color:#aaa;color:#fff;background-color:#f38630;}
    .tg .tg-1wig{font-weight:bold;text-align:left;vertical-align:top}
    .tg .tg-oo78{font-size:15px;color:#333333;border-color:#efefef;text-align:left;vertical-align:top}
    .tg .tg-l0rd{font-weight:bold;font-size:15px;color:#333333;border-color:#efefef;text-align:center;vertical-align:top}
    </style>"""

table_end = '</table>'


def printmd(string):
    if not isinstance(string, str):
        raise TypeError('The input parameter must be a string')
    display(Markdown(string))


def print_hbox(elements_list):
    if not isinstance(elements_list, list):
        raise TypeError('The input parameter must be a list')
    display(HBox(elements_list))


def table_header(headers):
    """
    Create the header of a table
    :param headers: list of strings for the headers or pandas dataframe,
                    from which the names of the columns will be taken
    :return:        html output
    Note:
        If the headers must include the string name that is already taken care of in the df_to_html table
    """
    if isinstance(headers, pd.core.frame.DataFrame):
        hdr = headers.columns
    elif isinstance(headers, list):
        hdr = headers
    else:
        raise TypeError('headers must be a list or pandas dataframe')

    t = ["""<th class="tg-1wig"><b>{}</th>""".format(h) for h in hdr]

    output = """
<table class="tg">
  <tr>
    {}
  </tr>
""".format(''.join(t))
    return output


def transform_row2html(new_row, index=True, decimals=8):
    """
    Create the html rows for a html table
    :param new_row:   list of values of the row
    :param index:     boolean. True if the index name needs to be printed
    :param decimals:  if any of the values is numeric, how many decimals are required in the html output
    :return:          html
    """
    if index:
        # Include the index in the row output
        new_row = [new_row.name, *new_row]
    try:
        r = ["""<td class="tg-oo78">{}</td>""".format(
            h if isinstance(h, str) else round(h, decimals)) for h in new_row]
    except:
        r = ["""<td class="tg-oo78">{}</td>""".format(h) for h in new_row]
    body = ''' <tr>
        {}
      </tr> '''.format(''.join(r))
    return body


def df_to_html_table(d, index=True, headers_replacement=None, decimals=8):
    """
    Creates a html table from a pandas dataframe
    :param d:         pandas dataframe (required)
    :param index:     boolean. True is the index needs to be printed. default is True
    :param headers_replacement: list of strings to replace the columns names (optional)
    :param decimals:  decimals to be printed for numeric values. Default 8
    :return:          html table
    """
    if not isinstance(d, pd.core.frame.DataFrame):
        raise TypeError("The parameter must be a pandas dataframe")

    # Create table headers
    if headers_replacement is not None:
        table_hdr = table_header(headers_replacement)
    else:
        if index:
            hdr_cols = [d.index.name, *list(d.columns)]
        else:
            hdr_cols = list(d.columns)
        table_hdr = table_header(hdr_cols)

    df_len = d.shape[0]
    table_body = ' '.join([transform_row2html(d.iloc[rown, :],
                                              index=index,
                                              decimals=decimals) for rown in range(df_len)])

    return table_style + table_hdr + table_body + table_end