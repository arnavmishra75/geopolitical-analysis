import google.generativeai as palm
import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from math import comb
import pprint
import matplotlib.pyplot as plt
import streamlit as st
asian_tensions = pd.read_csv("asian_tensions.csv")
palm.configure(api_key="key")
models = [m for m in palm.list_models()]
text_model = models[1].name
           
def create_top25_visual(df=asian_tensions):
    top_25 = df.sort_values(by='Tension Score', ascending=False).head(25)
    plt.figure(figsize=(12, 8))
    bars = plt.bar(top_25['Country Pairing'], top_25['Tension Score'])
    plt.xlabel('Country Pairing')
    plt.ylabel('Tension Score')
    plt.title('Top 25 Country Pairings with Highest Tension Scores')
    plt.xticks(rotation=45, ha='right')
    plt.xticks(fontsize=10)
    plt.yscale('log')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(plt)

def get_donut_chart(pair, df=asian_tensions, ax=None):
    tension_score = df[df["Country Pairing"]==pair]["Tension Score"].iloc[0]
    sizes = [tension_score, 100-tension_score]
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.figure
    my_circle = plt.Circle((0, 0), 1, color='white')
    ax.add_artist(my_circle)
    ax.pie(sizes, colors=['blue', 'white'], startangle=90, wedgeprops=dict(width=0.2))
    ax.axis('equal')
    center_text = f'{tension_score:.2f}'
    ax.text(0, 0, center_text+'%', ha='center', va='center', fontsize=24, color='black', fontweight='bold')
    ax.text(0, -1.35, pair, ha='center', va='center', fontsize=20, color='white', fontweight='bold')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

   
def plot_donut_charts(pairs):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(len(pairs)):
        get_donut_chart(pairs[i], ax=axs[i])
    plt.tight_layout()
    st.pyplot(fig, transparent=True)

def create_reco_buttons(pairs):
    for pair in pairs:
        button_label = "Receive AI Recommendations"
        button_key = f"{pair}_button"
        if st.button(button_label, key=button_key):
            ask_for_reco(pair.split('-')[0], pair.split('-')[1])

def ask_for_reco(country1, country2):
    prompt = f"What are the current political tensions between {country1} and {country2}? What are 3 steps they can take in order to bridge their differences and fix their issues? Please do not return a null answer and answer the question to the best of your abilities."

    with st.spinner("Generating AI Recommendations..."):
        completion = palm.generate_text(
            model=text_model,
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1200,
        )

    result = completion.result
    st.write(result)

def generate_ui(pairs):
    st.set_page_config(page_title='Harmony Amidst Hostility')                 
    st.markdown("<h1 style='text-align: center; color: blue;'>Harmony Amidst Hostility</h1>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center; color: white;'>Bridging Tense Geopolitical Relations in Asia through the Intelligence of Generative AI with Sentiment Analysis and Prompt Engineering</h6>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    create_top25_visual()
    plot_donut_charts(pairs)
    button_col1, button_col2, button_col3 = st.columns(3)
    with button_col1:
        create_reco_buttons(pairs[:1])
    with button_col2:
        create_reco_buttons(pairs[1:2])
    with button_col3:
        create_reco_buttons(pairs[2:])

selected_pairs=["Japan-N. Korea", "Saudi Arabia-Yemen", "Syria-Israel"]
generate_ui(selected_pairs)