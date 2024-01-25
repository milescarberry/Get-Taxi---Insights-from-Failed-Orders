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


		page_title="Gett - Insights from Failed Orders",


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


# HTML Para Function


def html_para(text = ''):


	# return f"<p style='text-align:center;line-height: 2.0;'>{text}</p>"


	return st.write(f"<p style='line-height: 2.0;'>{text}</p>", unsafe_allow_html = True)




# Dashboard Title
st.write(

		"<h1><center>Gett - Insights from Failed Orders</center></h1>",


		unsafe_allow_html=True


)


st.write("<br>", unsafe_allow_html=True)



# Some More Text


# st.write(

# 		"<h5><center>Some more text comes here.</center></h5>",

# 		unsafe_allow_html=True

# )


# st.write("<br>", unsafe_allow_html=True)




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



od_df = orders_df


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



# Creating Some Tabs


tab1, tab2, tab3 = st.tabs(

	['Dual Y-Axes Chart by Hour', 'Ride Cancellations Hexbin Map', 'Analysis']

)



# We will use these tabs later



# Creating Aggregated DataFrame By Hour


def get_aggregated_df(orders_df = orders_df):



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

	return _ 



_ = get_aggregated_df(orders_df = orders_df)



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


						line=dict(width=3.0)

				),


				secondary_y=False,


		)

		if st.session_state.right_y.lower().strip() != 'none':

			fig.add_trace(


					go.Scatter(


							x=_['time'],


							y=_[metric_mapper[st.session_state.right_y]],


							mode='lines',


							name=st.session_state.right_y,


							line=dict(width=3.0)

					),


					secondary_y=True,


			)

			fig.update_yaxes(

				title=st.session_state.right_y,

				secondary_y=True,

				showgrid=False

			)

		if len(_) > 16:

			fig.update_xaxes(title='Hour', tickangle=-45)

		else:

			fig.update_xaxes(title='Hour')

		fig.update_yaxes(title=st.session_state.left_y, secondary_y=False)

		if st.session_state.right_y.lower().strip() != 'none':

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

		else:

			hovertemp = ""

			if 'minutes' in st.session_state.left_y.lower():


				hovertemp = "<br><br>".join(

					[

						"<b>%{x}</b>",


						"<b>%{y:.1f} minutes</b><extra></extra>"

					]

				)



			elif 'offer' in st.session_state.left_y.lower():


				hovertemp = "<br><br>".join(

					[

						"<b>%{x}</b>",


						"<b>%{y} offers</b><extra></extra>"

					]

				)


			else:


				hovertemp = "<br><br>".join(

					[

						"<b>%{x}</b>",


						"<b>%{y} cancellations</b><extra></extra>"

					]

				)



			fig.update_traces(


				hovertemplate=hovertemp,


				selector=({'name': st.session_state.left_y})


			)







		if st.session_state.right_y.lower().strip() != 'none':


			fig.update_layout(


					title=dict(


							text=f"{st.session_state.left_y} & {st.session_state.right_y} by Hour",


							x=0.395,


							xanchor='center',


							yanchor='top',


							font=dict(size=23)


					)


			)


		else:


			fig.update_layout(


					title=dict(


							text=f"{st.session_state.left_y} by Hour",


							x=0.5,


							xanchor='center',


							yanchor='top',


							font=dict(size=23)


					)


			)




		fig.update_layout(

				# Alter the position of the legend.

				legend = {'x': 1.05, 'y': 1.0}

		)


		# fig.write_image("images/dual_y_axes.png")


		# if len(st.session_state.order_status) > 1:



		# 	query = f'''

		# 		Gett, previously known as GetTaxi, is an Israeli-developed technology platform solely focused on corporate Ground Transportation Management (GTM). 

		# 		They have an application where clients can order taxis, and drivers can accept their rides (offers). 

		# 		At the moment, when the client clicks the Order button in the application, the matching system searches for the most relevant drivers and offers them the order. 

		# 		I am actually investigating some matching metrics for orders that did not completed successfully, i.e., the customer didn't end up getting a car.

		# 		Please provide some valuable insights from this dual y-axes line chart where on the x-axis is the hour of the day, 

		# 		the left y-axis shows the {st.session_state.left_y} and its line color is closer to #f8776c and the right y-axis shows 

		# 		the {st.session_state.right_y} and its line color is closer to #a2a400. The x-axis ranges from {st.session_state.hours[0]} until {st.session_state.hours[1]}.

		# 		Please note that the ride can either be cancelled by the system i.e. the application or by the client. Please note that the number of cancellations by client is far greater than the number of cancellations by the system.

		# 		Also, the graph is based on the dataset that contains cancelled rides that had a driver assigned and vice versa.

		# 		Now, without any formalities directly proceed with extracting some valuable insights from this provided chart.


		# 	'''


		# bard_text = bard_api_response(query = query)


		return fig



with tab1:


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


	# charts = get_charts()


	st.plotly_chart(get_charts(), use_container_width = True)




with tab3:


	st.markdown("## Analysis")


	line_break()


	intro_text = "Gett, previously known as GetTaxi, is an Israeli-developed technology platform solely focused on corporate Ground Transportation Management (GTM).\nThey have an application where clients can order taxis, and drivers can accept their rides (offers).\nAt the moment, when the client clicks the Order button in the application, the matching system searches for the most relevant drivers and offers them the order.\nWe are investigating some matching metrics for orders that did not completed successfully, i.e. the customer didn't end up getting a car."


	html_para(text = intro_text)


	# st.markdown(

	# 	f"**{intro_text}**", 

	# 	unsafe_allow_html = True

	# )



	line_break()

	st.markdown("Let's take a look at this table: ")

	line_break()


	od_df = od_df.reset_index(drop = True)


	drop_indices = od_df[

			(od_df.order_status_key == 'Cancelled By System') & 

			(od_df.is_driver_assigned_key == 'Yes')

	].index.values 



	od_df = od_df[~od_df.index.isin(drop_indices)]


	od_df = od_df.reset_index(drop = True)


	grp_df = od_df.groupby(

		['is_driver_assigned_key', 'order_status_key'], 

		as_index = False, 

		dropna = False

		).agg(

		{

		"order_gk": pd.Series.nunique, 

		"cancellations_time_in_minutes": pd.Series.mean, 

		"m_order_eta": pd.Series.mean

		}


	)


	line_break()


	grp_df['%_of_total'] = grp_df.order_gk.apply(

		lambda x: x / len(orders_df)

	)


	grp_df = grp_df.reindex(

		columns = [

		'is_driver_assigned_key', 

		'order_status_key', 

		'order_gk', 

		'%_of_total', 

		'cancellations_time_in_minutes', 

		'm_order_eta'

		]

	)


	grp_df.cancellations_time_in_minutes = grp_df.cancellations_time_in_minutes.round(1)


	grp_df.m_order_eta = grp_df.m_order_eta.round(1)


	grp_df['%_of_total'] = grp_df['%_of_total'].apply(lambda x: f"{x:.1%}")


	grp_df.columns = ['Is Driver Assigned?', 'Order Status', 'Number of Orders', '% of Total Orders', 'Mean Cancellation Time in Minutes', 'Mean Time Before Order Arrival']



	st.dataframe(grp_df, use_container_width = True, hide_index = True)



	line_break()



	table_exp_text = '''

	The above table shows data on four types of orders:

	- Orders where no driver was assigned and the customer cancelled

	- Orders where no driver was assigned and the system cancelled

	- Orders where a driver was assigned and the customer cancelled

	- Orders where a driver was assigned and the system cancelled

	<br>

	Here are some key insights from the table:


	- **Customer cancellations are more common than system cancellations.** For both assigned and 
	unassigned driver orders, customer cancellations make up a larger percentage (68.2%) of the total cancelled orders.

	- **Orders where a driver was assigned are less likely to be cancelled.** The percentage of orders that are cancelled are
	lower when a driver is assigned (26.2%) compared to when no driver is assigned(73.8%).

	- **Cancellation times are longer for orders where a driver is assigned.** The mean cancellation time is more
	for orders where a driver is assigned (3.9 mins) compared to orders where a driver is not assigned (1.8 mins). Please note that the field **'Mean Cancellation Time In Minutes'**
	is applicable only for orders that are cancelled by the client.

	<br>

	**Please Note:** We will be dropping the instances where an order is cancelled by the system and a driver is assigned to that order
	since there are only 3 such instances.

	'''


	st.markdown(table_exp_text, unsafe_allow_html = True)



	# od_df = od_df[


	# 	(orders_df.order_status_key != 'Cancelled By System') & 


	# 	(orders_df.is_driver_assigned_key != 'Yes')


	# ]	



	# Distribution By Hour Plot


	line_break()


	line_break()


	st.markdown("### Ride Cancellations by Hour")


	line_break()


	num_orders_df_overall = od_df.groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(

			{"order_gk": pd.Series.nunique}

	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	num_orders_df_overall.columns = ['hour', 'cancelled_orders']





	num_orders_df_cl = od_df[od_df.order_status_key == 'Cancelled By Client'].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"order_gk": pd.Series.nunique}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	num_orders_df_cl.columns = ['hour', 'cancelled_orders']



	# st.write(num_orders_df_cl.shape)






	num_orders_df_sys = od_df[od_df.order_status_key == 'Cancelled By System'].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"order_gk": pd.Series.nunique}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	num_orders_df_sys.columns = ['hour', 'cancelled_orders']


	# st.write(num_orders_df_sys.shape)


	num_orders_df_cl_dr = od_df[


	(od_df.order_status_key == 'Cancelled By Client') & 


	(od_df.is_driver_assigned_key == 'Yes')

	].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"order_gk": pd.Series.nunique}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	num_orders_df_cl_dr.columns = ['hour', 'cancelled_orders']



	# st.write(num_orders_df_cl_dr.shape)





	num_orders_df_cl_no_dr = od_df[

		(od_df.order_status_key == 'Cancelled By Client') & 

		(od_df.is_driver_assigned_key == 'No')

	].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"order_gk": pd.Series.nunique}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	num_orders_df_cl_no_dr.columns = ['hour', 'cancelled_orders']



	# st.write(num_orders_df_cl_no_dr.shape)





	# Let's Plot an Overall Line Chart



	fig = go.Figure()



	fig.add_trace(


		go.Scatter(

			x = num_orders_df_overall.hour.apply(lambda x: str(x)), 

			y = num_orders_df_overall.cancelled_orders, 

			text = num_orders_df_overall.cancelled_orders, 

			mode = 'lines+markers',

			name = 'Ride Cancellations'

		)

	)



	hovertemp = "<br><br>".join(

		[

			"<b>Hour: %{x}</b>", 


			"<b>Ride Cancellations: %{y:,.0f}</b><extra></extra>"

		]

	)



	texttemp = "<b>%{y:,.0f}</b>"


	fig.update_traces(

		hovertemplate = hovertemp, 

		texttemplate = texttemp, 

		textposition = 'top center'

	)



	# fig.update_layout(

	# 	title = dict(


	# 		text = "Ride Cancellations by Hour\n", 


	# 		x = 0.5,


	# 		xanchor = 'center',


	# 		yanchor = 'top',


	# 		font=dict(size = 23)


	# 	)

	# )



	fig.update_xaxes(title = "Hour", showgrid = False)



	fig.update_yaxes(title = "Ride Cancellations")



	st.plotly_chart(fig, use_container_width = True)




	cancl_by_hour_text = '''


  From the above line chart we can deduce that,


  - The number of ride cancellations reaches a peak of 1,082 at 8 am.

  - There is a steady increase in the number of ride cancellations from 11 am to 5 pm.

  - There is drop in ride cancellations at 7 pm followed by a steep rise at 9 pm which lasts until 11 pm.

  - Additionally, we also noticed a growth in ride cancellations from 2 to 3 am.

  - The probable cause for the increase in ride cancellations at certain hours of the day: **Driver Demand-Supply Gap**
    
  - Clients cancel their rides when drivers aren't assigned quickly enough, or the system cancels rides if it can't find drivers within the set time limit.


	'''



	st.markdown(cancl_by_hour_text, unsafe_allow_html = True)




	line_break()


	line_break()



	st.markdown("### By Order Status")




	# Line Chart With Order Status Filters



	fig = go.Figure()



	fig.add_trace(

		go.Scatter(

			x = num_orders_df_cl.hour.apply(lambda x: str(x)), 

			y = num_orders_df_cl.cancelled_orders, 

			text = num_orders_df_cl.cancelled_orders, 

			mode = 'lines+markers',

			name = 'Cancelled by Client'

		)

	)



	fig.add_trace(

		go.Scatter(

			x = num_orders_df_sys.hour.apply(lambda x: str(x)), 

			y = num_orders_df_sys.cancelled_orders, 

			mode = 'lines+markers', 

			text = num_orders_df_sys.cancelled_orders,

			name = 'Cancelled by System'

		)

	)



	hovertemp = "<br><br>".join(

		[

			"<b>Hour: %{x}</b>", 


			"<b>Ride Cancellations: %{y:,.0f}</b><extra></extra>"

		]

	)



	texttemp = "<b>%{y:,.0f}</b>"


	fig.update_traces(

		hovertemplate = hovertemp, 

		texttemplate = texttemp, 

		textposition = 'top center'

	)



	# fig.update_layout(

	# 	title = dict(


	# 		text = "Ride Cancellations by Hour\n", 


	# 		x = 0.5,


	# 		xanchor = 'center',


	# 		yanchor = 'top',


	# 		font=dict(size = 23)


	# 	)

	# )



	fig.update_xaxes(title = "Hour", showgrid = False)



	fig.update_yaxes(title = "Ride Cancellations")



	st.plotly_chart(fig, use_container_width = True)



	line_break()



	cacl_by_status_text = '''

  - Apart from the fact that ride cancellations by client is greater than ride cancellations by system, **the trend is basically the same**.

  - There is a peak at 8 am, followed by a decline until 10 am and then a steady ascent from 11 am until 5 pm.

  - There is drop in ride cancellations at 7 pm followed by a steep rise at 9 pm which lasts until 11 pm.

  - There is also a growth in ride cancellations from 2 to 3 am.

  - The probable cause for the growth in ride cancellations at certain hours of the day remains the same as before: **Driver Demand-Supply Gap**

  - Clients cancel their rides when drivers aren't assigned quickly enough, or the system cancels rides if it can't find drivers within the set time limit.

  '''

 
	st.markdown(f"{cacl_by_status_text}", unsafe_allow_html = True)



	line_break()


	line_break()



	st.markdown('''### By Driver Assigned For Orders That Were Cancelled By Client''')




	# Line Chart With Driver Assigned Filters (For Orders That Were Cancelled By Client)



	fig = go.Figure()



	fig.add_trace(

		go.Scatter(

			x = num_orders_df_cl_dr.hour.apply(lambda x: str(x)), 

			y = num_orders_df_cl_dr.cancelled_orders, 

			text = num_orders_df_cl_dr.cancelled_orders, 

			mode = 'lines+markers',

			name = 'Driver Assigned'

		)

	)



	fig.add_trace(

		go.Scatter(

			x = num_orders_df_cl_no_dr.hour.apply(lambda x: str(x)), 

			y = num_orders_df_cl_no_dr.cancelled_orders, 

			mode = 'lines+markers', 

			text = num_orders_df_cl_no_dr.cancelled_orders,

			name = 'Driver Not Assigned'

		)

	)



	hovertemp = "<br><br>".join(

		[

			"<b>Hour: %{x}</b>", 


			"<b>Ride Cancellations: %{y:,.0f}</b><extra></extra>"

		]

	)



	texttemp = "<b>%{y:,.0f}</b>"


	fig.update_traces(

		hovertemplate = hovertemp, 

		texttemplate = texttemp, 

		textposition = 'top center'

	)



	# fig.update_layout(

	# 	title = dict(


	# 		text = "Ride Cancellations by Hour\n", 


	# 		x = 0.5,


	# 		xanchor = 'center',


	# 		yanchor = 'top',


	# 		font=dict(size = 23)


	# 	)

	# )



	fig.update_xaxes(title = "Hour", showgrid = False)



	fig.update_yaxes(title = "Ride Cancellations")



	st.plotly_chart(fig, use_container_width = True)


	line_break()


	dr_assigned_cacl_text = '''

  From the above chart we can deduce that,


  - Client ride cancellations where a driver was not assigned is greater than client ride cancellations where a driver was assigned.

  - In the former case, cancellations are the highest at 9 pm followed by 8 am and then at 11 pm. Also, we get to see a steady increase from 11 am till 5 pm. Additionally, we get to see an increase in cancellations from 2 to 3 am.

  - In the latter case too we get to see a spike at 8 am with a steady increase in cancellations from 11 am till 5 pm. We also get see a growth in cancellations from 10 pm to 11 pm and then a decline until 5 am.

  - We can draw a conclusion from the above chart that rides that have a driver assigned to them are less likely to get cancelled by the client and vice versa.

  - Long arrival times are likely the main reason clients cancel rides when a driver is assigned. They may not want to wait an excessive amount of time, especially when they are in a hurry.


	'''



	st.markdown(f"{dr_assigned_cacl_text}", unsafe_allow_html = True)



	line_break()


	line_break()


	st.markdown('### Mean Cancellation Time By Hour')



	cacl_time_df_cl_dr = od_df[


	(od_df.order_status_key == 'Cancelled By Client') & 


	(od_df.is_driver_assigned_key == 'Yes')

	].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"cancellations_time_in_minutes": pd.Series.mean}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	cacl_time_df_cl_dr.columns = ['hour', 'mean_cancellations_time_in_minutes']



	# st.write(num_orders_df_cl_dr.shape)



	cacl_time_df_cl_no_dr = od_df[

		(od_df.order_status_key == 'Cancelled By Client') & 

		(od_df.is_driver_assigned_key == 'No')

	].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"cancellations_time_in_minutes": pd.Series.mean}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	cacl_time_df_cl_no_dr.columns = ['hour', 'mean_cancellations_time_in_minutes']




	# Mean Cancellations Time Line Chart


	fig = go.Figure()



	fig.add_trace(

		go.Scatter(

			x = cacl_time_df_cl_dr.hour.apply(lambda x: str(x)), 

			y = cacl_time_df_cl_dr.mean_cancellations_time_in_minutes, 

			text = cacl_time_df_cl_dr.mean_cancellations_time_in_minutes, 

			mode = 'lines+markers',

			name = 'Driver Assigned'

		)

	)



	fig.add_trace(

		go.Scatter(

			x = cacl_time_df_cl_no_dr.hour.apply(lambda x: str(x)), 

			y = cacl_time_df_cl_no_dr.mean_cancellations_time_in_minutes, 

			mode = 'lines+markers', 

			text = cacl_time_df_cl_no_dr.mean_cancellations_time_in_minutes,

			name = 'Driver Not Assigned'

		)

	)



	hovertemp = "<br><br>".join(

		[

			"<b>Hour: %{x}</b>", 


			"<b>%{y:.1f} minutes</b><extra></extra>"

		]

	)



	texttemp = "<b>%{y:.1f}</b>"



	fig.update_traces(

		hovertemplate = hovertemp, 

		texttemplate = texttemp, 

		textposition = 'top center'

	)



	# fig.update_layout(

	# 	title = dict(


	# 		text = "Ride Cancellations by Hour\n", 


	# 		x = 0.5,


	# 		xanchor = 'center',


	# 		yanchor = 'top',


	# 		font=dict(size = 23)


	# 	)

	# )



	fig.update_xaxes(title = "Hour", showgrid = False)



	fig.update_yaxes(title = "Mean Cancellation Time in Minutes")



	st.plotly_chart(fig, use_container_width = True)


	line_break()


	cacl_time_text = '''
  
  From the above chart we can deduce that,


  - The mean ride cancellation time by client when a driver is assigned is greater than the mean ride cancellation time when a driver is not assigned.


  - When a driver is assigned, clients might hold off on canceling because they hope the driver will arrive soon. Conversely, when a driver isn't readily available, clients might cancel more quickly due to uncertainty about the wait time. Ultimately, the system's processing time plays a role in both scenarios.


  - The mean cancellation time is the maximum at 5 am and is the lowest at hours 7 and 8 in cases where a driver is assigned. This is followed by a peak at 11 am then a decline until 3 pm and again an incline until 6 pm. The mean cancellation time dips again at 7 pm followed by a steady increase until 11 pm. This is followed by peaks until 3 am.


  - In the case of ride cancellations where a driver is not assigned, the mean cancellation time is the maximum at 4 pm followed by peaks at 11 pm and from 5 am to 9 am. It reaches its lowest point at 10am and then gradually rises hitting a maximum at 4pm.


	'''


	st.markdown(f"{cacl_time_text}", unsafe_allow_html = True)



	line_break()


	line_break()



	st.markdown("### Mean Time Before Order Arrival By Hour")




	# DataFrame



	eta_time_df_cl = od_df[


	(od_df.order_status_key == 'Cancelled By Client')

	].groupby(

			['booking_hour'],

			as_index=False,

			dropna=False

	).agg(


			{"m_order_eta": pd.Series.mean}


	).sort_values(


			by=['booking_hour'],


			ascending=[True]


	)



	eta_time_df_cl.columns = ['hour', 'mean_m_order_eta']




	# Mean Time Before Order Arrival In Minutes By Hour Line Chart



	fig = go.Figure()



	fig.add_trace(

		go.Scatter(

			x = eta_time_df_cl.hour.apply(lambda x: str(x)), 

			y = eta_time_df_cl.mean_m_order_eta, 

			text = eta_time_df_cl.mean_m_order_eta, 

			mode = 'lines+markers',

			name = 'Mean Time Before Order Arrival in Minutes'

		)

	)





	hovertemp = "<br><br>".join(

		[

			"<b>Hour: %{x}</b>", 


			"<b>%{y:.1f} minutes</b><extra></extra>"

		]

	)



	texttemp = "<b>%{y:.1f} minutes</b>"


	fig.update_traces(

		hovertemplate = hovertemp, 

		texttemplate = texttemp, 

		textposition = 'top center'

	)



	# fig.update_layout(

	# 	title = dict(


	# 		text = "Ride Cancellations by Hour\n", 


	# 		x = 0.5,


	# 		xanchor = 'center',


	# 		yanchor = 'top',


	# 		font=dict(size = 23)


	# 	)

	# )



	fig.update_xaxes(title = "Hour", showgrid = False)



	fig.update_yaxes(title = "Mean Time Before Order Arrival in Minutes")



	st.plotly_chart(fig, use_container_width = True)



	line_break()


	m_order_eta_text = '''

	From the above chart we can deduce that,


  - The time before order arrival peaks at 8 am with an acute drop until 10 am. Thereafter it gradually increases until 5 pm followed by an acute drop until 8 pm. Thereafter it again rises until 11 pm.

  - The peaks at 8 am and 5 pm likely coincide with morning and evening rush hours, when road traffic significantly increases.


	'''



	st.markdown(f"{m_order_eta_text}", unsafe_allow_html = True)







	# # Plotly Histogram Function


	# def get_kde_plotly(df, colname):

	# 		# Booking Hour KDE Plot

	# 		# fig = go.Figure()


	# 		# hist_data = [df[colname]]


	# 		# group_labels = ['distplot']


	# 		# fig = ff.create_distplot(

	# 		# 	hist_data = hist_data, 

	# 		# 	group_labels = group_labels, 

	# 		# 	curve_type='kde',

	# 		# 	show_rug = False

	# 		# )


	# 		# add the kernel density estimate plot

	# 		# fig.add_trace(


	# 		#     go.Histogram(


	# 		#         x=df[colname],


	# 		#         histnorm='probability',


	# 		#         name=f'{colname} KDE Plot'


	# 		#         # text = df[colname]


	# 		#     )


	# 		# )


	# 		fig = exp.histogram(

	# 			df, 

	# 			x = colname, 

	# 			color = 'is_driver_assigned_key',

	# 			histnorm = 'probability',

	# 			# nbins = 26,

	# 			cumulative = False

	# 		)


	# 		fig.update_xaxes(title=f'{colname}')


	# 		fig.update_yaxes(title='Probability')


	# 		texttemp = "<b>%{y:.3f}</b>"


	# 		hovertemp = "<br><br>".join(


	# 				[

	# 						f"<b>{colname}: </b>" + "<b>%{x}</b>",


	# 						"<b>%{y:.3f}</b><extra></extra>"
	# 				]


	# 		)


	# 		fig.update_traces(


	# 				hovertemplate=hovertemp


	# 				# texttemplate = texttemp,


	# 				# textposition = 'outside'


	# 		)


	# 		# fig.update_traces(

	# 		# 	bar = dict(

	# 		# 		color = 'Blue'

	# 		# 		)

	# 		# )


	# 		st.plotly_chart(fig, use_container_width = True)

	# 		# fig.show()


	# get_kde_plotly(ords_df, 'booking_hour')




with tab2:


	# Hexbin Plotly Maps


	# Set Mapbox Access Token


	exp.set_mapbox_access_token(st.secrets["mapbox_access_token"])



	# Preparing DataFrame 


	coord_grp = 	ords_df.groupby(

				['origin_longitude', 'origin_latitude'], 

				as_index  = False, 

				dropna = False


		).agg(

			{"order_gk": pd.Series.nunique}


		).sort_values(

			by = ['order_gk'], 

			ascending = False

		).reset_index(drop = True)



	coord_grp.order_gk = coord_grp.order_gk.apply(

		lambda x: 0 if 'nan' in str(x).lower() else x

		)



	coord_grp.columns = ['origin_longitude', 'origin_latitude', 'cancellations']


	coord_grp['cancellations_cumsum'] = coord_grp.cancellations.cumsum()


	coord_grp['%_of_all_orders'] = coord_grp.cancellations_cumsum.apply(

		lambda x: round(x / len(ords_df), 2)

		)



	coord_grp = coord_grp.reset_index(drop = True)



	# st.write(len(coord_grp[coord_grp['%_of_all_orders'] <= 0.8]))



	# Hexbin Map


	fig = ff.create_hexbin_mapbox(

		data_frame = coord_grp, 

		lat = 'origin_latitude', 

		lon = 'origin_longitude', 

		nx_hexagon = 8,

		opacity = 0.70,

		labels = {"color": 'Ride Cancellations'},

		min_count = 1,

		color_continuous_scale = exp.colors.sequential.Oranges

		# show_original_data = True,

		# original_data_marker = dict(size = 6, opacity = 0.7, color = 'deeppink')


	)


	hovertemp = '<b>Ride Cancellations = %{z:,.0f}</b><extra></extra>'



	fig.update_traces(hovertemplate = hovertemp)



	fig.update_layout(mapbox_style= "carto-positron")



	fig.update_layout(


		title = dict(


				text = "Ride Cancellations Hexbin Map", 


				x = 0.43,


				xanchor = 'center',


				yanchor = 'top',


				font=dict(size = 23)

		

			)


		)



	fig.update_layout(height = 500)



	st.plotly_chart(fig, use_container_width = True)





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
