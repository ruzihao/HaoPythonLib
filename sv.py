import os
import sys
import sweetviz as sv
import pandas as pd
import sweetviz.sv_html as sv_html
import webbrowser
from sweetviz.dataframe_report import DataframeReport


infile = sys.argv[1]
outfile = sys.argv[2]


class DataframeReportHZ(DataframeReport):
    def __init__(self, *args, **kwargs):
        super(DataframeReportHZ, self).__init__(*args, **kwargs)

    def show_html(self, filepath='SWEETVIZ_REPORT.html', layout='widescreen'):
        sv_html.load_layout_globals_from_config()
        self.page_layout = layout
        sv_html.set_summary_positions(self)
        sv_html.generate_html_detail(self)
        self._page_html = sv_html.generate_html_dataframe_page(self)

        # self.temp_folder = config["Files"].get("temp_folder")
        # os.makedirs(os.path.normpath(self.temp_folder), exist_ok=True)

        f = open(filepath, 'w', encoding="utf-8-sig")
        f.write(self._page_html)
        f.close()

        # Not sure how to work around this: not fatal but annoying...
        # https://bugs.python.org/issue5993
        webbrowser.open('file://' + os.path.realpath(filepath))


df = pd.read_csv(infile)
rep = DataframeReportHZ(df.dropna(how='all', axis=1), pairwise_analysis='off')
rep.show_html(outfile)