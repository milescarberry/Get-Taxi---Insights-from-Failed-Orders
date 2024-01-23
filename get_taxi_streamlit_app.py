import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


from plotly import express as exp, graph_objects as go, figure_factory as ff, io as pio


from plotly.subplots import make_subplots


from pandas_utils.pandas_utils_2 import *


# import ipywidgets as widgets

# from IPython.display import display

import warnings


import streamlit as st


from streamlit_autorefresh import st_autorefresh


import datetime as dt


st.set_page_config(


    page_title="Page Title",


    layout='wide'


)


warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

pio.templates.default = 'ggplot2'

sns.set_style('darkgrid')

sns.set_context('paper', font_scale=1.4)


# HTML Line Break Function


def line_break():

    return st.write("<br>", unsafe_allow_html=True)


# Dashboard Title
st.write(

    "<h1><center>Title</center></h1>",


    unsafe_allow_html=True


)


st.write("<br>", unsafe_allow_html=True)


# Some More Text


st.write(

    "<h5><center>Some more text comes here.</center></h5>",

    unsafe_allow_html=True

)


st.write("<br>", unsafe_allow_html=True)


# Get Data

@st.cache_data
def get_data():

    datasets = pd.read_csv(
        "./datasets/data_offers.csv"), pd.read_csv("./datasets/data_orders.csv")

    offers_df = datasets[0]

    orders_df = datasets[1]

    # Sidebar Filters

    # with st.sidebar:

    # 	# Some Dataset Filters

    # Do your work here

    # In[2]:

    # st.write(offers_df.shape, orders_df.shape)

    # st.dataframe(orders_df.sample(3))

    orders_df['order_status_key'] = orders_df['order_status_key'].apply(

        lambda x: 'Cancelled By Client' if x == 4 else 'Cancelled By System'


    )

    orders_df['cancellations_time_in_seconds'] = orders_df.cancellations_time_in_seconds.apply(

        lambda x: x / 60

    )

    orders_df = orders_df.rename(


        {'cancellations_time_in_seconds': 'cancellations_time_in_minutes'},


        axis=1


    )

    orders_df.m_order_eta = orders_df.m_order_eta.apply(

        lambda x: x / 60


    )

    # st.dataframe(pd.DataFrame(
    #     orders_df.order_status_key.value_counts() / len(orders_df)))

    # st.write("\n")

    # show_nan(orders_df)

    # st.write("\n")

    # st.dataframe(offers_df.sample(3))

    # st.write("\n")

    offers_cnt_df = offers_df.groupby(

        ['order_gk'],

        as_index=False, dropna=False

    ).agg(

        {"offer_id": pd.Series.nunique}

    ).sort_values(

        by=['order_gk'],

        ascending=[True]

    )

    offers_cnt_df = offers_cnt_df.rename(

        {"offer_id": 'num_offers_applied'}, axis=1

    )

    st.write("\n")

    orders_df = pd.merge(orders_df, offers_cnt_df,
                         on='order_gk', how='left')

    orders_df.num_offers_applied = orders_df.num_offers_applied.apply(

        lambda x: 0 if 'nan' in str(x).lower() else x

    )

    # st.dataframe(orders_df.sample(5))

    # st.write("\n")

    # grp_df = orders_df.groupby(

    #     ['order_status_key', 'is_driver_assigned_key'],

    #     as_index=False,

    #     dropna=False

    # ).agg(

    #     {

    #         "cancellations_time_in_minutes": pd.Series.mean,

    #         "m_order_eta": pd.Series.mean,

    #         'order_gk': pd.Series.nunique,

    #         'num_offers_applied': pd.Series.mean

    #     }

    # ).sort_values(by=[

    #     'order_status_key',

    #     'is_driver_assigned_key'

    # ], ascending=[

    #     True,

    #     True

    # ]

    # )

    # grp_df.columns = [

    #     'order_status_key',

    #     'is_driver_assigned_key',

    #     'mean_cancellations_time_in_mins',

    #     'mean_m_order_eta_mins',

    #     'num_orders',

    #     'mean_num_offers_applied'

    # ]

    # grp_df['%_of_total_orders'] = grp_df['num_orders'] / len(orders_df)

    # grp_df = grp_df.reindex(columns=[

    #     'order_status_key',

    #     'is_driver_assigned_key',

    #     'mean_cancellations_time_in_mins',

    #     'mean_m_order_eta_mins',

    #     '%_of_total_orders',

    #     'num_orders',

    #     'mean_num_offers_applied'

    # ])

    # grp_df.mean_num_offers_applied = grp_df.mean_num_offers_applied.apply(

    #     lambda x: round(x, 0)

    # )

    # st.dataframe(grp_df)

    orders_df.is_driver_assigned_key = orders_df.is_driver_assigned_key.apply(


        lambda x: 'Yes' if x == 1 else 'No'


    )

    # st.write("<br><br><br>", unsafe_allow_html=True)

    # st.write("m_order_eta only when a driver is assigned.")

    # st.write("\n")

    # st.write("cancellations_time_in_minutes only when cancelled by client.")

    # st.write("<br>", unsafe_allow_html=True)

    # st.write(

    #     "<br><br>Filters:<br><br>1.Hour of Ride Booking<br><br>2.orders_status_key<br><br>3.is_driver_assigned_key",

    #     unsafe_allow_html=True

    # )

    # st.write("<br>", unsafe_allow_html=True)

    # st.write(

    #     "Metrics: <br><br>1.Count Orders<br><br>2.Mean Cancellation Time In Minutes <br><br>3.Mean m_order_eta in Minutes<br><br>4.Mean num_offers_applied<br><br>5. Most Frequently Used offer_id",

    #     unsafe_allow_html=True

    # )

    # st.write("<br>", unsafe_allow_html=True)

    # st.dataframe(pd.DataFrame(orders_df.dtypes))

    # st.write("<br>", unsafe_allow_html=True)

    orders_df['order_datetime'] = orders_df.order_datetime.apply(

        lambda x: dt.datetime.strptime(x, "%H:%M:%S")

    )

    orders_df['booking_hour'] = orders_df.order_datetime.apply(

        lambda x: int(dt.datetime.strftime(x, "%H")

                      )


    )

    orders_df['booking_minute'] = orders_df.order_datetime.apply(

        lambda x: int(dt.datetime.strftime(x, "%M"))


    )

    orders_df = orders_df.reindex(


        columns=[
            "order_datetime",
            "booking_hour",
            "booking_minute",
            "origin_longitude",
            "origin_latitude",
            "m_order_eta",
            "order_gk",
            "order_status_key",
            "is_driver_assigned_key",
            "cancellations_time_in_minutes",
            "num_offers_applied"
        ]



    )

    return orders_df, offers_df


orders_df, offers_df = get_data()


# Sidebar Filters


with st.sidebar:

    st.write("<h1><center>Filters</center></h1>", unsafe_allow_html=True)

    line_break()

    # Creating Several Session State Variables

    if 'hours' not in st.session_state:

        st.session_state.hours = (dt.time(8, 0), dt.time(18, 0))

    if 'order_status' not in st.session_state:

        order_statuses = orders_df.order_status_key.unique().tolist()

        order_statuses.sort()

        st.session_state.order_status = order_statuses

    if 'is_driver_assigned' not in st.session_state:

        is_driver_keys = ['Yes', 'No']

        st.session_state.is_driver_assigned = is_driver_keys

  # Hours Slider

    def change_hours_sel():

        st.session_state.hours = st.session_state.new_hours

    hours_sel = st.slider(


        "Time Period",


        value=(dt.time(8, 0), dt.time(18, 0)),


        on_change=change_hours_sel,


        key='new_hours'


    )

    line_break()

    line_break()

    # Order Status Multi Select

    order_status_list = ['All']

    unique_order_status = orders_df.order_status_key.unique().tolist()

    unique_order_status.sort()

    order_status_list.extend(unique_order_status)

    def change_order_status_sel():

        st.session_state.order_status = st.session_state.new_order_status

        options = orders_df.order_status_key.unique().tolist()

        options.sort()

        if 'All' in st.session_state.order_status:

            st.session_state.order_status = options

        elif len(st.session_state.order_status) == 0:

            st.session_state.order_status = options

        else:

            pass

    order_status_sel = st.multiselect(

        "Order Status",

        order_status_list,

        ['All'],


        on_change=change_order_status_sel,


        key='new_order_status'


    )

    line_break()

    line_break()

    def change_driver_assigned_sel():

        st.session_state.is_driver_assigned = st.session_state.new_is_driver_assigned

        keys = ['Yes', 'No']

        if 'All' in st.session_state.is_driver_assigned:

            st.session_state.is_driver_assigned = keys

        elif len(st.session_state.is_driver_assigned) == 0:

            st.session_state.is_driver_assigned = keys

        else:

            pass

    driver_assigned_sel = st.multiselect(


        "Driver Assigned",


        ['All', 'Yes', 'No'],


        ['All'],


        on_change=change_driver_assigned_sel,


        key='new_is_driver_assigned'



    )


# Applying Filters To DataFrame


orders_df = orders_df[

    (orders_df.order_status_key.isin(st.session_state.order_status)) &


    (orders_df.is_driver_assigned_key.isin(
        st.session_state.is_driver_assigned))


]


# Creating Some Top Line Metrics


ords_df = orders_df[


    (orders_df.booking_hour >= st.session_state.hours[0].hour) &


    (orders_df.booking_hour <= st.session_state.hours[1].hour)


]


# Top Line KPIs


kpicol1, kpicol2, kpicol3 = st.columns(3)


with kpicol1:

    with st.container(border=True):

        # Mean m_order_eta

        m_order_eta_kpi = st.metric(

            "Mean Time Before Order Arrival",

            f"{ords_df.m_order_eta.mean():.1f} Mins"


        )


with kpicol2:

    with st.container(border=True):

        # Mean cancellations_time_in_minutes

        cancellation_time_kpi = st.metric(

            "Mean Cancellation Time",

            f"{ords_df.cancellations_time_in_minutes.mean():.1f} Mins"


        )


with kpicol3:

    with st.container(border=True):

        # Mean num_offers_applied

        offers_applied_kpi = st.metric(

            "Mean Offers Applied Per Order",

            f"{ords_df.num_offers_applied.mean():.0f} Offers"


        )


line_break()


# Creating Aggregated DataFrame By Hour


hours = []


minutes = []


for i in range(0, 24):

    for j in range(0, 60):

        hours.append(i)

        minutes.append(j)


_ = pd.DataFrame()


_['hour'] = hours


_['minute'] = minutes


_avg_m_order_eta = []


_avg_cancellation_minutes = []


_sum_cancellations = []


_avg_num_offers_applied = []


for i in range(len(_)):

    mean_m_order_eta = orders_df[


        (orders_df.booking_hour == _.iloc[i, ::1][0]) &


        (orders_df.booking_minute == _.iloc[i, ::1][1])

    ].m_order_eta.mean()

    mean_cancellations_time = orders_df[


        (orders_df.booking_hour == _.iloc[i, ::1][0]) &


        (orders_df.booking_minute == _.iloc[i, ::1][1])


    ].cancellations_time_in_minutes.mean()

    mean_num_offers_applied = orders_df[


        (orders_df.booking_hour == _.iloc[i, ::1][0]) &


        (orders_df.booking_minute == _.iloc[i, ::1][1])


    ].num_offers_applied.mean()

    sum_order_cancellations = orders_df[


        (orders_df.booking_hour == _.iloc[i, ::1][0]) &


        (orders_df.booking_minute == _.iloc[i, ::1][1])


    ].order_gk.nunique()

    _avg_m_order_eta.append(mean_m_order_eta)

    _avg_cancellation_minutes.append(mean_cancellations_time)

    _avg_num_offers_applied.append(mean_num_offers_applied)

    _sum_cancellations.append(sum_order_cancellations)


_['mean_m_order_eta'] = _avg_m_order_eta


_['mean_cancellations_time_in_mins'] = _avg_cancellation_minutes


_['mean_num_offers_applied'] = _avg_num_offers_applied


_['total_order_cancellations'] = _sum_cancellations


# Applying Hours Filter To _ DataFrame


st_time = st.session_state.hours[0]

en_time = st.session_state.hours[1]


_ = _[

    (_.hour >= st_time.hour) &


    (_.hour <= en_time.hour)

]


_ = _.reset_index(drop=True)


_['time'] = _.apply(lambda x: dt.time(int(x[0]), int(x[1])), axis=1)


_ = _.reindex(columns=[

    "time",
    "hour",
    "minute",
    "mean_m_order_eta",
    "mean_cancellations_time_in_mins",
    "mean_num_offers_applied",
    "total_order_cancellations"

]


)


_ = _.groupby(['hour'], as_index=False, dropna=False).agg(

    {

        "mean_m_order_eta": pd.Series.mean,

        "mean_cancellations_time_in_mins": pd.Series.mean,

        "mean_num_offers_applied": pd.Series.mean,

        "total_order_cancellations": pd.Series.sum

    }


)


_.mean_num_offers_applied = _.mean_num_offers_applied.round(0)


_['time'] = _['hour'].apply(lambda x: dt.datetime.strftime(


    dt.datetime(1990, 6, 1, int(x), 0),


    "%H:%M"


)

)


# Get Secondary Y-Axis Charts Function


def get_charts():


    # Secondary Y-Axis Charts


    fig = make_subplots(specs=[[{"secondary_y": True}]])


    fig.add_trace(


        go.Scatter(


            x=_['time'],


            y=_[metric_mapper[st.session_state.left_y]],


            mode='lines',


            name=st.session_state.left_y,


            line = dict(width = 3.0)

        ),


        secondary_y=False,


    )



    fig.add_trace(


        go.Scatter(


            x=_['time'],


            y=_[metric_mapper[st.session_state.right_y]],


            mode='lines',


            name=st.session_state.right_y,


            line = dict(width = 3.0)

        ),


        secondary_y=True,


    )

    fig.update_yaxes(title=st.session_state.left_y, secondary_y=False)

    fig.update_yaxes(title=st.session_state.right_y,
                     secondary_y=True, showgrid=False)



    if len(_) > 16:


    	fig.update_xaxes(title='Hour', tickangle = -45)


    else:


    	fig.update_xaxes(title = 'Hour')



    if 'minutes' in st.session_state.right_y.lower() and 'minutes' in st.session_state.left_y.lower():

        hovertemp = "<br><br>".join(

            [

                "<b>%{x}</b>",

                "<b>%{y:.1f} minutes</b><extra></extra>",

                ""

            ]

        )

        fig.update_traces(


            hovertemplate=hovertemp,


            selector=({'name': st.session_state.right_y})


        )

        fig.update_traces(


            hovertemplate=hovertemp,


            selector=({'name': st.session_state.left_y})


        )

    elif 'minutes' in st.session_state.right_y.lower():

        hovertemp_1 = "<br><br>".join(

            [

                "<b>%{x}</b>",

                "<b>%{y:.1f} minutes</b><extra></extra>"

            ]

        )

        if 'offer' in st.session_state.left_y.lower():

            hovertemp_2 = "<br><br>".join(

                [

                    "<b>%{x}</b>",


                    "<b>%{y} offers</b><extra></extra>"

                ]

            )

        else:

            hovertemp_2 = "<br><br>".join(

                [

                    "<b>%{x}</b>",


                    "<b>%{y} cancellations</b><extra></extra>"

                ]

            )

        fig.update_traces(


            hovertemplate=hovertemp_1,


            selector=({'name': st.session_state.right_y})


        )

        fig.update_traces(


            hovertemplate=hovertemp_2,


            selector=({'name': st.session_state.left_y})


        )

    elif 'minutes' in st.session_state.left_y.lower():

        hovertemp_1 = "<br><br>".join(

            [

                "<b>%{x}</b>",


                "<b>%{y:.1f} minutes</b><extra></extra>"

            ]

        )

        if 'offer' in st.session_state.right_y.lower():

            hovertemp_2 = "<br><br>".join(

                [

                    "<b>%{x}</b>",


                    "<b>%{y} offers</b><extra></extra>"

                ]

            )

        else:

            hovertemp_2 = "<br><br>".join(

                [

                    "<b>%{x}</b>",


                    "<b>%{y} cancellations</b><extra></extra>"

                ]

            )

        fig.update_traces(


            hovertemplate=hovertemp_1,


            selector=({'name': st.session_state.left_y})


        )

        fig.update_traces(


            hovertemplate=hovertemp_2,


            selector=({'name': st.session_state.right_y})


        )

    else:

        pass

    fig.update_layout(


        title=dict(


            text=f"{st.session_state.left_y} & {st.session_state.right_y} by Hour",


            x=0.5,


            xanchor='center',


            yanchor='top',


            font=dict(size=23)


        )


    )

    fig.update_layout(

        # Alter the position of the legend.
        legend={'x': 1.05, 'y': 1.0}

    )

    return fig


# Filters For Secondary Y-Axis Charts


metric_mapper = {


    "Number of Order Cancellations": 'total_order_cancellations',


    "Mean Cancellation Time in Minutes": 'mean_cancellations_time_in_mins',


    "Mean Time Before Order Arrival in Minutes": 'mean_m_order_eta',


    "Mean Offers Applied": "mean_num_offers_applied"


}


portray_metrics = list(metric_mapper.keys())


# Choose Y-Axes


if 'left_y' not in st.session_state:

    st.session_state.left_y = 'Number of Order Cancellations'


if 'right_y' not in st.session_state:

    st.session_state.right_y = "Mean Cancellation Time in Minutes"


if 'portray_metrics_left' not in st.session_state:

    st.session_state.portray_metrics_left = [

        i for i in portray_metrics if i != st.session_state.right_y
    ]


if 'portray_metrics_right' not in st.session_state:


    st.session_state.portray_metrics_right = [

        i for i in portray_metrics if i != st.session_state.left_y

    ]


    st.session_state.portray_metrics_right.extend(["None"])



# Two Columns for Y-Axes Metrics


ycol1, ycol2 = st.columns(2)


with ycol1:

    def change_left_y_sel():

        if st.session_state.new_left_y != st.session_state.new_right_y:

            st.session_state.left_y = st.session_state.new_left_y

        else:

            st.session_state.new_left_y = st.session_state.left_y

            st.toast(

                f"Left and Right Y-axes cannot be the same.",

                icon='ℹ️'

            )

    left_y_sel = st.selectbox(


        "Choose Metric for Left Y-Axis",


        st.session_state.portray_metrics_left,


        index=0,


        on_change=change_left_y_sel,


        key='new_left_y'


    )


with ycol2:

    def change_right_y_sel():

        if st.session_state.new_right_y != st.session_state.new_left_y:

            st.session_state.right_y = st.session_state.new_right_y

        else:

            st.session_state.new_right_y = st.session_state.right_y

            st.toast(

                f"Left and Right Y-axes cannot be the same.",

                icon='ℹ️'

            )

    right_y_sel = st.selectbox(


        "Choose Metric for Right Y-Axis",


        st.session_state.portray_metrics_right,


        index=0,


        on_change=change_right_y_sel,


        key='new_right_y'


    )


line_break()


# Display Secondary Y-Axis Plotly Line Chart


st.plotly_chart(get_charts(), use_container_width=True)


line_break()


# Some KDE Plots


line_break()


num_orders_df = ords_df.groupby(

    ['booking_hour'],

    as_index=False,

    dropna=False

).agg(

    {"order_gk": pd.Series.nunique}

).sort_values(


    by=['booking_hour'],


    ascending=[True]


)


# num_orders_df['rolling_mean'] = num_orders_df.order_gk.rolling(

#     window=2,

#     center=True

# ).mean()


# Num Cancelled Orders By Hour Bar Chart


fig = exp.line(


    num_orders_df,


    x='booking_hour',


    y='order_gk',


    text='order_gk'


)


texttemp = "<b>%{text:.0f}</b>"


hovertemp = "<br><br>".join([


    "<b>Hour: %{x}</b>",


    "<b>Cancelled Orders: %{y:.0f}</b><extra></extra>"

]

)


fig.update_traces(

    texttemplate=texttemp,

    textposition='top center',

    hovertemplate=hovertemp

)


fig.update_xaxes(title='Hour')


fig.update_yaxes(title='Cancelled Orders')


# fig.update_layout(bargap = 0.32)


st.plotly_chart(fig, use_container_width=True)


line_break()


# Plotly Histogram Function


def get_kde_plotly(df, colname):

    # Booking Hour KDE Plot

    # fig = go.Figure()


    # hist_data = [df[colname]]


    # group_labels = ['distplot']


    # fig = ff.create_distplot(

    # 	hist_data = hist_data, 

    # 	group_labels = group_labels, 

    # 	curve_type='kde',

    # 	show_rug = False

    # )


    # add the kernel density estimate plot

    # fig.add_trace(


    #     go.Histogram(


    #         x=df[colname],


    #         histnorm='probability',


    #         name=f'{colname} KDE Plot'


    #         # text = df[colname]


    #     )


    # )


    fig = exp.histogram(

    	df, 

    	x = colname, 

    	color = 'is_driver_assigned_key',

    	histnorm = 'probability',

    	# nbins = 26,

    	cumulative = False

    )


    fig.update_xaxes(title=f'{colname}')


    fig.update_yaxes(title='Probability')


    texttemp = "<b>%{y:.3f}</b>"


    hovertemp = "<br><br>".join(


        [

        		f"<b>{colname}: </b>" + "<b>%{x}</b>",


            "<b>%{y:.3f}</b><extra></extra>"
        ]


    )


    fig.update_traces(


        hovertemplate=hovertemp


        # texttemplate = texttemp,


        # textposition = 'outside'


    )


    # fig.update_traces(

    # 	bar = dict(

    # 		color = 'Blue'

    # 		)

    # )


    st.plotly_chart(fig, use_container_width = True)

    # fig.show()


get_kde_plotly(ords_df, 'booking_hour')




# CSS Styling Of Elements


def styling_func():

    css = '''


		div[class^='st-emotion-cache-16txtl3'] { 


		 padding-top: 1rem; 


		}


		div[class^='block-container'] { 

		  padding-top: 1rem; 

		}


		[data-testid="stMetric"] {
		    width: fit-content;
		    margin: auto;
		}

		[data-testid="stMetric"] > div {
		    width: fit-content;
		    margin: auto;
		}

		[data-testid="stMetric"] label {
		    width: fit-content;
		    margin: auto;
		}


		[data-testid="stMarkdownContainer"] > p {

          font-weight: bold;

        }


        [data-testid="stMetricValue"] {

          font-weight: bold;
          
        }


	'''

    st.write(


        f"<style>{css}</style>",


        unsafe_allow_html=True


    )


styling_func()


# Footer Section


# Mention Data Source


# st.write("<br><br><br><br>", unsafe_allow_html = True)


# st.write(

# 	'''<footer class="css-164nlkn egzxvld1"><center><p>Data Source: <a href="https://data.telangana.gov.in/" target="_blank" class="css-1vbd788 egzxvld2">data.telangana.gov.in</a></p></center></footer>''',


# 	unsafe_allow_html = True


# 	)
