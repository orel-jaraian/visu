import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import random
import plotly.express as px
from scipy.stats import t
import numpy as np
from streamlit_plotly_events import plotly_events

# Load your dataset
# Load your dataset
data = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

scatter_features = ['HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity', 'HvyAlcoholConsump',
                    'NoDocbcCost', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

# Age range mapping
age_mapping = {
    1: '18-24',
    2: '25-29',
    3: '30-34',
    4: '35-39',
    5: '40-44',
    6: '45-49',
    7: '50-54',
    8: '55-59',
    9: '60-64',
    10: '65-69',
    11: '70-74',
    12: '75-79',
    13: '80-99'
}

colors = [
    "#FBE9E7", "#FFCCBC", "#FFAB91", "#FF8A65", "#FF7043",
    "#FF5722", "#F4511E", "#E64A19", "#D84315", "#BF360C"
]
cmap = LinearSegmentedColormap.from_list("deep_orange_cmap", colors, N=256)

# Compute correlations with 'Diabetes_binary'
correlation_diabetes = {feature: data[feature].corr(data['Diabetes_binary']) for feature in scatter_features}
correlation_heartdisease = {feature: data[feature].corr(data['HeartDiseaseorAttack']) for feature in scatter_features}
correlation_highbp = {feature: data[feature].corr(data['HighBP']) for feature in scatter_features}
correlation_stroke = {feature: data[feature].corr(data['Stroke']) for feature in scatter_features}

# st.markdown(
#     """
#     <style>
#     .centered-title {
#         font-family: Arial, sans-serif;
#         color: black;
#         font-size: 40px;
#         font-weight: bold;
#     }
#     .select-box-label {
#         font-family: Arial, sans-serif;
#         font-size: 20px;
#         font-weight: bold;
#         margin-top: 30px;
#     }
#     </style>
#     <div class="centered-title">Diabetes - Analysis of Risk Factors</div>
#     """,
#     unsafe_allow_html=True
# )
#
# # Use st.write to add the label for the radio button
# st.write('<p class="select-box-label">Select a page: 📖</p>', unsafe_allow_html=True)
# page = st.radio("", ['Lifestyle Indicators', "clustermap - Income & Education",
#                      'Correlations with other diseases', 'BMI & Age Affect on Diabetes',
#                      'The Effects of Mental & Physical Health On Diabetes'])
st.sidebar.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-family: Arial, sans-serif;
        color: black;
        font-size: 24px;
        font-weight: bold;
    }
    .select-box-label {
        font-family: Arial, sans-serif;
        font-size: 16px;
        font-weight: bold;
        margin-top: 30px;  /* Adjust this value to move the label down */

    }
    </style>
    <div class="centered-title">Diabetes - Analysis of Risk Factors</div>
    """,
    unsafe_allow_html=True
)

# Apply custom CSS class to the select box label
st.sidebar.markdown('<p class="select-box-label">Select a page: 📖</p>', unsafe_allow_html=True)
page = st.sidebar.selectbox("", ['Lifestyle Indicators', "Clustermap - Income & Education",
                                 'Correlations With Other Diseases', 'BMI & Age Affect on Diabetes', 'Mental & Physical Health'])


if page == 'Lifestyle Indicators':
    data_copy = data.rename(columns={
    'Smoker': 'Smoker',
    'PhysActivity': 'Physical Activity',
    'Fruits': 'Eating Fruits',
    'Veggies': 'Eating Vegetables',
    'HvyAlcoholConsump': 'Heavy Alcohol Drinkers'
})

    features = ['Smoker', 'Physical Activity', 'Eating Fruit', 'Eating Vegetables', 'Heavy Alcohol Drinkers']


    st.markdown(
        "<h1 style='text-align: center; font-family: Arial; color: black;'>Lifestyle Indicators</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <style>
    .arial-black-text {
        font-family: Arial, sans-serif;
        color: black;
    }
    </style>
    <div class="arial-black-text">
        One of the key goals in preventing diabetes is identifying the leading risk factors for the disease.<br>
        To this end, there is an organization called the Behavioral Risk Factor Surveillance System
        that conducted a comprehensive survey in 2015 among 253,680 respondents.<br><br>
        <b>Select one of the lifestyle features from the filter below to view the differences in diabetes prevalence rates.
        In addition, selection of an age range and gender will allow you to view the distribution of diabetes prevalence rates.</b>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns([1, 3])

    with col1:
        # Add a selectbox for feature selection
        st.markdown(
            "<div style='font-size: 15px; font-family: Arial; color: black; margin-top: 120px;'><b>Select Feature: 👇</b></div>",
            unsafe_allow_html=True)
        selected_feature = st.selectbox('', features)

        # Add a slider for age range selection
        st.markdown(
            "<div style='font-size: 15px; font-family: Arial; color: black; margin-top: 30px;'><b>Select Age Category:👴</b></div>",
            unsafe_allow_html=True)
        age_range = st.slider('', min_value=1, max_value=13, value=(1, 13), step=1, format='%d')

        # Add a multi-select for sex selection
        st.markdown(
            "<div style='font-size: 15px; font-family: Arial; color: black; margin-top: 30px;'><b>Select Gender: 👨👩</b></div>",
            unsafe_allow_html=True)
        sex_selection = st.multiselect('', ['Women', 'Men'], default=['Women', 'Men'])

    # Map sex selection to filter values
    sex_map = {'Women': 0, 'Men': 1}

    # Display the corresponding age range and gender
    selected_age_range = f"<b style='font-family: Arial; color: black;'>Selected Age Range: <u>{age_mapping[age_range[0]][:2]} - {age_mapping[age_range[1]][3:5]}</u></b>"
    selected_gender = f"<b style='font-family: Arial; color: black;'>Selected Gender: <u>{', '.join(sex_selection)}</u></b>"

    with col2:
        # Calculate percentages for the selected feature, age range, and gender
        def calculate_percentage(feature, age_range_scale=None, sex=None):
            print(age_range_scale)
            filtered_data = data_copy
            if age_range_scale:
                filtered_data = filtered_data[filtered_data['Age'].between(age_range_scale[0], age_range_scale[1])]
            if sex:
                filtered_data = filtered_data[filtered_data['Sex'].isin([sex_map[s] for s in sex])]
            percentages = filtered_data.groupby(feature)['Diabetes_binary'].mean() * 100
            percentages.index = percentages.index.map({0: 'No', 1: 'Yes'})
            return percentages


        percentages = calculate_percentage(selected_feature, age_range, sex_selection)

        # Create the bar plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=percentages.index,
            y=percentages,
            marker_color=[('blue' if val == 'Yes' else 'orange') for val in percentages.index],
            text=[f"<b>{height:.2f}%<b>" for height in percentages],
            textposition='outside',
            hoverinfo='x+y'
        ))
        fig.update_layout(
            title={
                'text': f'{selected_feature} vs. Percent of Diabetes<br>(Ages: {age_mapping[age_range[0]][:2]}-{age_mapping[age_range[1]][3:5]} & Gender: {", ".join(sex_selection)})',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20, 'color': 'black', 'family': 'Arial'}
            },

            xaxis=dict(
                title=dict(
                    text=f'<b>{selected_feature}</b>',
                    font=dict(
                        family='Arial',
                        size=18,
                        color='black'
                    )
                )
            ),
            yaxis=dict(
                title=dict(
                    text='<b>Percent of Diabetes</b>',
                    font=dict(
                        family='Arial',
                        size=18,
                        color='black'
                    )
                )
            ),
            font=dict(size=20, family='Arial', color='black'),  # Adjust font size for better readability
            height=700,  # Increase plot height
            width=1500,  # Increase plot width
            margin=dict(l=50, r=50)  # Move plot to the right by adjusting left and right margins
        )
        fig.update_traces(textfont_size=14)  # Increase font size for text on bars

        # Display the Plotly plot
        st.plotly_chart(fig)


elif page == "Clustermap - Income & Education":
    st.markdown("""
    <style>
    .arial-black {
        font-family: Arial, sans-serif;
        color: black;
    }
    .centered-text {
        text-align: center;
    }
    </style>
    <h1 class="arial-black centered-text">Heatmap showing clustering based on Income & Education</h1>
    <p class="arial-black">
    Background characteristics such as education level and income level are factors that can influence the likelihood of developing diabetes.\n
     We used clustering to identify which levels show significant differences and a heatmap to detect trends in disease rates across different combinations.
    </p>
    """, unsafe_allow_html=True)

    # Pivot the DataFrame for the heatmap
    pivot_table = pd.pivot_table(data, values='Diabetes_binary',
                                 index='Income', columns='Education',
                                 aggfunc='mean')

    # Create a buffer to save the plot
    buffer = BytesIO()

    clustermap = sns.clustermap(pivot_table,
                                cmap=cmap,
                                annot=True,
                                fmt='.2f',
                                annot_kws={"size": 14},
                                linewidths=0.5,
                                linecolor='black',
                                cbar_kws={'label': 'Mean Diabetes Rate'},
                                cbar_pos=(0.87, 0.2, 0.05, 0.4))
    clustermap.cax.yaxis.label.set_size(13)

    clustermap.ax_heatmap.set_position([0.2, 0.1, 0.6, 0.6])
    clustermap.ax_row_dendrogram.set_position([0.1, 0.1, 0.1, 0.6])
    clustermap.ax_col_dendrogram.set_position([0.2, 0.7, 0.6, 0.1])

    # Set font size for the axis labels
    clustermap.ax_heatmap.set_xlabel('Education', fontsize=16)
    clustermap.ax_heatmap.set_ylabel('Income', fontsize=16)
    plt.setp(clustermap.ax_heatmap.xaxis.get_majorticklabels(), fontsize=14)
    plt.setp(clustermap.ax_heatmap.yaxis.get_majorticklabels(), fontsize=14)

    plt.savefig(buffer, format='png')
    plt.close()

    buffer.seek(0)

    # Display the clustermap
    st.image(buffer, use_column_width=True)


elif page == 'Correlations With Other Diseases':
    data_copy = data.rename(columns={
    'HeartDiseaseorAttack': 'Heart Disease or Attack',
    })
    st.markdown(
        "<h1 style='text-align: center;font-size: 43px;'>How different health and behavioral characteristics affect diseases?</h1>",
        unsafe_allow_html=True)
    st.markdown("""
        <style>
        .arial-black {
            font-family: Arial, sans-serif;
            color: black;
        }
        </style>
        <div class="arial-black">
            Clinical studies have shown that diabetes increases the risk of developing other diseases, such as heart disease or stroke. 
            The graph illustrates how different health and behavioral characteristics either strengthen or weaken the risk of diabetes in conjunction with another disease.
            Select a disease from the list in the control below to discover the common risk factors.
        </div>
        """, unsafe_allow_html=True)
    # Add a selectbox for selecting the target feature
    st.markdown(
        "<div style='font-size: 18px;'><b>Select a target feature to analyze correlations with Diabetes:🧑‍⚕️</b></div>",
        unsafe_allow_html=True)
    target_feature = st.selectbox('', ['Heart Disease or Attack', 'High Blood Pressure', 'Stroke'])

    # Prepare the data for the selected target feature
    if target_feature == 'Heart Disease or Attack':
        correlations = correlation_heartdisease
        ylabel = 'Correlation with Heart Disease or Attack'
        plot_data = pd.DataFrame({
            'Feature': scatter_features,
            'Correlation_Diabetes': [correlation_diabetes[feature] for feature in scatter_features],
            'Correlation_Target': [correlations[feature] for feature in scatter_features]
        })
    elif target_feature == 'High Blood Pressure':
        correlations = correlation_highbp
        ylabel = 'Correlation with High Blood Pressure'
        plot_data = pd.DataFrame({
            'Feature': [feature for feature in scatter_features if feature != 'HighBP'],
            'Correlation_Diabetes': [correlation_diabetes[feature] for feature in scatter_features if
                                     feature != 'HighBP'],
            'Correlation_Target': [correlations[feature] for feature in scatter_features if feature != 'HighBP']
        })
    elif target_feature == 'Stroke':
        correlations = correlation_stroke
        ylabel = 'Correlation with Stroke'
        plot_data = pd.DataFrame({
            'Feature': scatter_features,
            'Correlation_Diabetes': [correlation_diabetes[feature] for feature in scatter_features],
            'Correlation_Target': [correlations[feature] for feature in scatter_features]
        })

    # Create the scatter plot using Plotly
    fig = go.Figure()

    # Generate a list of random colors
    num_points = len(plot_data)
    random_colors = ['rgb({},{},{})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for
                     _ in range(num_points)]

    fig.add_trace(go.Scatter(
        x=plot_data['Correlation_Diabetes'],
        y=plot_data['Correlation_Target'],
        mode='markers+text',
        text=plot_data['Feature'],
        textposition='top center',
        marker=dict(size=14, color=random_colors, line=dict(width=2, color='Black')),
        hovertemplate='<b>%{text}</b><br>Correlation with Diabetes: %{x:.2f}<br>Correlation with Target: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': f'Correlation with Diabetes and {target_feature}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'black', 'family': 'Arial'}

        },
        xaxis_title={
            'text': 'Correlation with Diabetes',
            'font': {'size': 18, 'color': 'black', 'family': 'Arial'},
        },
        yaxis_title={
            'text': ylabel,
            'font': {'size': 18, 'color': 'black', 'family': 'Arial'},
        },
        xaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            range=[plot_data['Correlation_Diabetes'].min() - 0.1, plot_data['Correlation_Diabetes'].max() + 0.1],
            tickfont={'size': 15, 'weight': 'bold', 'family': 'Arial'}
        ),
        yaxis=dict(
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            range=[plot_data['Correlation_Target'].min() - 0.1, plot_data['Correlation_Target'].max() + 0.1],
            tickfont={'size': 15, 'weight': 'bold', 'family': 'Arial'}
        ),
        height=700,  # Increase the plot height
        width=1000  # Increase the plot width
    )

    # Display the plot
    st.plotly_chart(fig)

elif page == 'BMI & Age Affect on Diabetes':
    st.markdown(
        "<h1 style='text-align: center; font-family: Arial; color: black;'>The Relationship Between Diabetes, Age, and BMI</h1>",
        unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Arial; color: black;'>
    In the previous graphs, we concluded that age and BMI are key parameters in diagnosing diabetes. Here, we will delve into identifying trends in their relationships.
    The graph presents the average for each BMI and age group along with a 95% confidence interval. Additionally, filters for gender and smoking populations can be applied.
    <br><br>
    <b>The conclusion is that across all age ranges and filters examined, individuals with diabetes have significantly higher BMI.</b>
    </div>
    """, unsafe_allow_html=True)

    # Map binary diabetes values to readable strings
    data['Diabetes_binary'] = data['Diabetes_binary'].map({0: 'No Diabetes', 1: 'Diabetes'})

    # Sidebar filters with predefined lists
    sex_options = ['Both', 'Male', 'Female']
    smoker_options = ['Both', 'Smokers', 'Non-smokers']

    # Create a container for the filters
    filter_container = st.container()
    with filter_container:
        col1, col2 = st.columns(2)  # Two columns for filters

        with col1:
            st.markdown(
                "<div style='font-family: Arial; font-size: 17px; font-weight: bold;margin-top: 30px;'>Select Sex 👨👩</div>",
                unsafe_allow_html=True)
            sex_filter = st.selectbox('', sex_options)

        with col2:
            st.markdown(
                "<div style='font-family: Arial; font-size: 17px; font-weight: bold;margin-top: 30px;'>Smoking🚬:</div>",
                unsafe_allow_html=True)
            smoker_filter = st.selectbox('', smoker_options)

    # Filter data based on selections
    filtered_data = data.copy()

    if sex_filter != 'Both':
        sex_value = 1 if sex_filter == 'Male' else 0
        filtered_data = filtered_data[filtered_data['Sex'] == sex_value]

    if smoker_filter != 'Both':
        smoker_value = 1 if smoker_filter == 'Smokers' else 0
        filtered_data = filtered_data[filtered_data['Smoker'] == smoker_value]

    filtered_data['Age'] = filtered_data['Age'].map(age_mapping)

    # Ensure the order of x-axis labels
    ordered_age_groups = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
                          '70-74', '75-79', '80-99']

    # Aggregate data to calculate mean BMI and standard error for each age group and diabetes category
    agg_data = filtered_data.groupby(['Age', 'Diabetes_binary']).agg(
        mean_BMI=('BMI', 'mean'),
        std_BMI=('BMI', 'std'),
        count=('BMI', 'count')
    ).reset_index()

    # Calculate the standard error of the mean (SEM)
    agg_data['sem_BMI'] = agg_data['std_BMI'] / np.sqrt(agg_data['count'])

    # Calculate the confidence interval (95%)
    confidence_level = 0.95
    degrees_freedom = agg_data['count'] - 1
    confidence_interval = t.ppf((1 + confidence_level) / 2., degrees_freedom) * agg_data['sem_BMI']
    agg_data['ci_upper'] = agg_data['mean_BMI'] + confidence_interval
    agg_data['ci_lower'] = agg_data['mean_BMI'] - confidence_interval

    # Prepare the plot title with a line break
    plot_title = f"Line plot of BMI vs Age<br>(Sex: {sex_filter}, Smoking: {smoker_filter})"

    # Create a Plotly line plot with hover information
    fig = go.Figure()

    # Add traces for each Diabetes category
    colors = {'Diabetes': 'orange', 'No Diabetes': 'blue'}
    fillcolors = {'Diabetes': 'rgba(255,165,0,0.2)', 'No Diabetes': 'rgba(0,0,255,0.3)'}

    for diabetes_status in agg_data['Diabetes_binary'].unique():
        df = agg_data[agg_data['Diabetes_binary'] == diabetes_status]

        # Add the shaded area for confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([df['Age'], df['Age'][::-1]]),
            y=pd.concat([df['ci_upper'], df['ci_lower'][::-1]]),
            fill='toself',
            fillcolor=fillcolors[diabetes_status],
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

        # Add the line plot with markers
        fig.add_trace(go.Scatter(
            x=df['Age'],
            y=df['mean_BMI'],
            mode='lines+markers',
            name=diabetes_status,
            line=dict(color=colors[diabetes_status]),
            hovertemplate='<b>Age:</b> %{x}<br><b>BMI:</b> %{y:.2f}<br><b>Upper CI:</b> %{customdata[0]:.2f}<br><b>Lower CI:</b> %{customdata[1]:.2f}<br><b>Status:</b> ' + diabetes_status,
            customdata=np.stack((df['ci_upper'], df['ci_lower']), axis=-1)
        ))

    # Update layout for better readability
    fig.update_layout(
        title={
            'text': plot_title,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 28, 'color': 'black', 'family': 'Arial'}
        },
        xaxis=dict(
            title=dict(
                text='<b>Age</b>',
                font=dict(
                    family='Arial',
                    size=18,
                    color='black'
                )
            ),
            tickmode='array',
            tickvals=ordered_age_groups,
            ticktext=ordered_age_groups,
            tickangle=38,
            tickfont=dict(
                family='Arial',
                size=15,
                color='black'
            )
        ),
        yaxis=dict(
            title=dict(
                text='<b>Mean BMI</b>',
                font=dict(
                    family='Arial',
                    size=18,
                    color='black'
                )
            ),
            tickfont=dict(
                family='Arial',
                size=15,
                color='black'
            )
        ),
        font=dict(size=20, family='Arial', color='black'),
        height=700,
        width=1500,
        margin=dict(l=50, r=50)
    )

    # Display the Plotly plot
    st.plotly_chart(fig)

elif page == 'Mental & Physical Health':
    st.markdown(
        """
        <style>
        .arial-black {
            font-family: Arial, sans-serif;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True)

    # Main title in the center with the custom style
    st.markdown(
        "<h1 class='arial-black' style='text-align: center;'>Cross-Filtering Dashboard - Distribution of Mental Health</h1>",
        unsafe_allow_html=True)

    # Description with the custom style
    st.markdown("""<div class='arial-black'>
        In these graphs, we can see the 5 statuses of General Health and look side-by-side at the distributions of Mental Health and Physical Health across each General Health status.
        <br>
        <b>The bar plots show how the number of days with a poor state of mental/physical health out of the last month affects the number of positive patients with Diabetes in the filtered data.<b>     
        <br><br>
        </div>
        """,
                unsafe_allow_html=True)

    filtered_data = data.copy()
    # Mapping of labels to their meanings
    label_meanings = {
        1: 'Excellent',
        2: 'Very Good',
        3: 'Good',
        4: 'Fair',
        5: 'Poor'
    }

    # Create the donut chart for general health distribution
    genhlth_totals = filtered_data['GenHlth'].value_counts().sort_index(ascending=False)  # Sort categories from 5 to 1

    # Colors with different luminance and saturation 
    colors = ['#e0440e', '#e6693e', '#ec8f6e', '#f3b49f', '#f6c7b6']

    # Map the labels to their meanings
    labels_with_meanings = [f"{int(label)} - {label_meanings[label]}" for label in genhlth_totals.index]

    fig_donut = go.Figure(data=[go.Pie(
        labels=labels_with_meanings,
        values=genhlth_totals,
        hole=0.45,  # Adjusted to match the example
        textinfo='percent',
        insidetextorientation='radial',
        marker=dict(colors=colors),
        sort=False  # Ensure the order of categories is preserved
    )])
    fig_donut.update_layout(
        title_text='Diabetes Patients by General Health Status',
        font=dict(
            family='Arial',
            size=16,
            color='black'
        )
    )

    # Display donut chart and capture click events
    selected = plotly_events(fig_donut, select_event=True, click_event=True)

    # Filter data based on selected general health status
    selected_genhlth = None
    if selected:
        selected_genhlth = selected[0]['pointNumber']

    if selected_genhlth is not None:
        selected_genhlth_value = genhlth_totals.index[selected_genhlth]
        filtered_data = filtered_data[filtered_data['Diabetes_binary'] == 1]  # Filter only for rows where Diabetes_binary is 1
        filtered_data = filtered_data[filtered_data['GenHlth'] == selected_genhlth_value]
        st.markdown(
            f"<span style='font-size: 17px; font-weight: bold;font-family:Arial;'>Filtering by General Health Status Value: <u>{int(selected_genhlth_value)}</u></span>",
            unsafe_allow_html=True
        )
    else:
        filtered_data = data.copy()

    # Define variables for the bar plots after filtering
    menthlth_diabetes = filtered_data[filtered_data['Diabetes_binary'] == 1]['MentHlth']
    physhlth_diabetes = filtered_data[filtered_data['Diabetes_binary'] == 1]['PhysHlth']

    # Create a range of days from 1 to 29
    days_range = np.arange(1, 30)

    # Ensure all days from 1 to 29 are included in the plots
    menthlth_counts = menthlth_diabetes.value_counts().reindex(days_range, fill_value=0)
    physhlth_counts = physhlth_diabetes.value_counts().reindex(days_range, fill_value=0)

    # Display the bar plots below the donut chart
    st.markdown("<h2 style='text-align: center;'>Mental and Physical Health Distributions</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        # Bar plot for Mental Health (blue)
        fig_bar_ment = px.bar(
            x=menthlth_counts.index,
            y=menthlth_counts.values,
            labels={'x': 'Days of Poor Mental Health', 'y': 'Number of Diabetes Patients'},
            title='Distribution of Poor Mental Health Days for Diabetes Patients',
            color_discrete_sequence=['blue']
        )
        fig_bar_ment.update_layout(
            font=dict(
                family="Arial, sans-serif",
                size=15,
                color="black",
                weight="bold"
            ),
            title=dict(
                text='The distribution of Poor Mental<br>Health Days for Diabetes Patients',
                x=0.22,  # Center the title
                font=dict(
                    family="Arial, sans-serif",
                    size=15,
                    color="black",
                    weight="bold")
            ),
            xaxis=dict(
                title=dict(
                    font=dict(
                        family="Arial, sans-serif",
                        size=16,
                        color="black",
                        weight="bold"
                    )
                ),
                tickfont=dict(
                    family="Arial, sans-serif",
                    size=16,
                    color="black",
                    weight="bold"
                )
            ),
            yaxis=dict(
                title=dict(
                    font=dict(
                        family="Arial, sans-serif",
                        size=16,
                        color="black",
                        weight="bold"
                    )
                ),
                tickfont=dict(
                    family="Arial, sans-serif",
                    size=16,
                    color="black",
                    weight="bold"
                )
            )
        )
        st.plotly_chart(fig_bar_ment)

    with col2:
        # Bar plot for Physical Health (orange)
        fig_bar_phys = px.bar(
            x=physhlth_counts.index,
            y=physhlth_counts.values,
            labels={'x': 'Days of Poor Physical Health', 'y': 'Number of Diabetes Patients'},
            title='The distribution of Poor Physical<br>Health Days for Diabetes Patients',
            color_discrete_sequence=['orange']
        )
        fig_bar_phys.update_layout(
            font=dict(
                family="Arial, sans-serif",
                size=15,
                color="black",
                weight="bold"
            ),
            title=dict(
                text='The distribution of Poor Physical<br>Health Days for Diabetes Patients',
                x=0.22,  # Center the title
                font=dict(
                    family="Arial, sans-serif",
                    size=15,
                    color="black",
                    weight="bold"
                )
            ),
            xaxis=dict(
                title=dict(
                    font=dict(
                        family="Arial, sans-serif",
                        size=16,
                        color="black",
                        weight="bold"
                    )
                ),
                tickfont=dict(
                    family="Arial, sans-serif",
                    size=16,
                    color="black",
                    weight="bold"
                )
            ),
            yaxis=dict(
                title=dict(
                    font=dict(
                        family="Arial, sans-serif",
                        size=16,
                        color="black",
                        weight="bold"
                    )
                ),
                tickfont=dict(
                    family="Arial, sans-serif",
                    size=16,
                    color="black",
                    weight="bold"
                )
            )
        )
        st.plotly_chart(fig_bar_phys)



