import streamlit as st
import pickle 
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn

def load_model():
    with open('cancer.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
rf = data['model']
s = data['s']

st.subheader(":blue[Get your condition tested!]")

clump = st.slider("Enter the clump thickness",2,6,2)#st.number_input("Enter the clump thickness")
unifsize = st.slider("Enter the uniformity of cell size",0,5,0)#st.number_input("Enter the uniformity of cell size")
margadh = st.slider("Enter the marginal adhesion",0,4,0)#st.number_input("Enter the marginal adhesion")
singepisize = st.slider("Enter the size of single epithelial cell",2,4,2)#st.number_input("Enter the size of single epithelial cell")
barnuc = st.slider("Presence of bare nuclei in the cell",0,5,0)#st.number_input("Presence of bare nuclei in the cell")
blandchrom = st.slider("Enter the bland chromatin",2,5,0)#st.number_input("Enter the bland chromatin")
normnuclei = st.slider("Enter the normality of nuclei",0,4,0)#st.number_input("Enter the normality of nuclei")


predict = st.button("Predict")

if predict:
    x = np.array([[clump,unifsize,margadh,singepisize,barnuc,blandchrom,normnuclei]])
    x = s.transform(x)
    y = rf.predict(x)

    if y[0]==4:
        st.subheader(":red[Danger! You may have early signs of Cancer.]")
        st.subheader("We suggest you to fix an appointment with your doctor.")

    else:
        st.balloons()
        st.subheader(":green[Congratulations!!!]")
        st.subheader(":green[As per your details, you seem to be out of danger.]")
        st.write("Still, we suggest you to follow an anti-cancer lifestyle to remain safe from this life threatening disease.")
        
