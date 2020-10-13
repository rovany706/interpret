from ..api.base import ExplainerMixin, ExplanationMixin
from ..utils import unify_data, gen_name_from_class, gen_local_selector

import pysubgroup as ps
import pandas as pd
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import dash_html_components as html
import dash_table as dt

class SubgroupDiscoveryExplainer(ExplainerMixin):
    """ Provides subgroup discovery task """

    available_explanations = ["data"]
    explainer_type = "data"

    def __init__(self):
        """ Initializes class.
        """
        return

    def explain_data(
        self,
        data,
        target_name,
        result_set_size=5,
        depth=2,
        qf=ps.WRAccQF(),
        name=None
    ):
        """ Explains data using Subgroup Discovery task.

        Args:
            data: Data to explain.
            target_name: Name of a target variable
            result_set_size: Number of resulting subgroups
            depth: Maximum number of selectors combined in a subgroup description
            qf: Candidtate scoring measure
            name: User-defined explanation name.

        Returns:
            An explanation object.
        """

        if name is None:
            name = gen_name_from_class(self)

        result_dicts = []
        for target_value in [True, False]:
            target = ps.BinaryTarget(target_name, target_value)
            searchspace = ps.create_selectors(data, ignore=[target_name])
            task = ps.SubgroupDiscoveryTask(
                data, 
                target, 
                searchspace, 
                result_set_size=result_set_size, 
                depth=depth, 
                qf=qf
            )

            result = ps.BeamSearch().execute(task)
            result_dict = {
                "name": f'{target_name} = {target_value}',
                "data": data,
                "result": result,
                "result_dataframe": result.to_dataframe()
            }
            result_dicts.append(result_dict)


        internal_obj = {
            "overall": None,
            "specific": result_dicts,
        }

        selector_columns = ['Name']
        records = []
        for result_dict in result_dicts:
            record = {}
            record['Name'] = result_dict['name']
            records.append(record)
    
        selector = pd.DataFrame.from_records(records, columns=selector_columns)

        return SubgroupDiscoveryExplanation(
            "data",
            internal_obj,
            selector=selector
        )

class SubgroupDiscoveryExplanation(ExplanationMixin):
    """ Explanation object specific to subgroup discovery explainer."""

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        name=None,
        selector=None,
    ):
        """ Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """

        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.name = name
        self.selector = selector

    def data(self, key=None):
        """ Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.

        Returns:
            A serializable dictionary.
        """
        if key is None:
            return self._internal_obj['overall']

        specific_dict = self._internal_obj["specific"][key].copy()
        return specific_dict

    def visualize(self, key=None):
        result_dict = self.data(key)
        if result_dict is None: # if None generate tables
            html_elements = []
            for d in self._internal_obj["specific"]:
                html_elements.append(html.H3(d['name']))
                table = self.generate_table(d['result_dataframe'][['quality', 'subgroup']])
                html_elements.append(table)
                
            return html.Div(html_elements)

        # generate plots
        plot_sgbars = ps.plot_sgbars(result_dict['result_dataframe'], result_dict['data'])
        img_sgbars = self.fig_to_uri(plot_sgbars)
        plot_npspace = ps.plot_npspace(result_dict['result_dataframe'], result_dict['data'])
        img_npspace = self.fig_to_uri(plot_npspace)
        plot_similarity_dendrogram = ps.similarity_dendrogram(result_dict['result'], result_dict['data'])
        img_similarity_dendrogram = self.fig_to_uri(plot_similarity_dendrogram)

        html_out = f'''
        <img src={img_sgbars} />
        <br />
        <img src={img_npspace} />
        <br />
        <img src={img_similarity_dendrogram} />
        '''
        return html_out

    def fig_to_uri(self, in_fig, close_all=True, **save_args):
        # type: (plt.Figure) -> str
        """
        Save a figure as a URI
        :param in_fig:
        :return:
        """
        out_img = BytesIO()
        in_fig.savefig(out_img, format='png', **save_args)
        if close_all:
            in_fig.clf()
            plt.close('all')
        out_img.seek(0)  # rewind file
        encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)

    def generate_table(self, dataframe):
        records = dataframe.to_dict("records")
        columns = [
            {"name": col, "id": col}
            for _, col in enumerate(dataframe.columns)
            if col != "id"
        ]
        output_table = dt.DataTable(
                    data=records,
                    columns=columns,
                    filter_action="native",
                    sort_action="native",
                    editable=False,
                    id="overall-graph-{0}".format(0),
                )
        
        return output_table

