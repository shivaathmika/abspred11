import streamlit as st
import streamlit_ketcher
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
import requests, joblib, io
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
from mordred import Calculator
from mordred import ExtendedTopochemicalAtom, AcidBase, Aromatic, AtomCount, BondCount
from mordred import CarbonTypes, Constitutional, EState, HydrogenBond, Lipinski, PathCount, Polarizability
from mordred import RingCount, RotatableBond, SLogP, TopoPSA, Weight

def load_joblib_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    return joblib.load(io.BytesIO(r.content))



def leverage_calculator(data1: pd.DataFrame, data2: pd.DataFrame):
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("Input files must have the same number of columns/features.")
    
    p_val = len(data1.columns)
    n_val = len(data1)
    h_star = 3 * (p_val + 1) / n_val
    #print(h_star)

    # Work on copies to avoid modifying the original DataFrames
    data1_ = data1.copy()
    data2_ = data2.copy()

    data1_.insert(0, "dummey", 1)
    data2_.insert(0, "dummey", 1)

    X = data1_.values  # shape (n, p)
    X_test = data2_.values  # shape (m, p)

    # (X^T X)^-1
    XtX_inv = np.linalg.inv(X.T @ X)

    # Leverage for training data
    H_train = X @ XtX_inv @ X.T
    leverage_tr = np.diag(H_train)

    # Leverage for test data
    leverage_te = np.array([x @ XtX_inv @ x.T for x in X_test])

    leverage_tr_df = pd.DataFrame(leverage_tr, columns=["Leverage Value"], index=data1.index)
    leverage_te_df = pd.DataFrame(leverage_te, columns=["Leverage Value"], index=data2.index)

    leverage_tr_df["AD Status"] = np.where(leverage_tr_df["Leverage Value"] >= h_star, "Outside AD", "Inside AD")
    leverage_te_df["AD Status"] = np.where(leverage_te_df["Leverage Value"] >= h_star, "Outside AD", "Inside AD")

    return leverage_te_df

def smiles_to_pil(smiles: str, size=(340, 280)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.drawOptions().padding = 0.12
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        bio = io.BytesIO(drawer.GetDrawingText())
        return Image.open(bio).convert("RGBA")
    except Exception:
        return None

def process_smiles(smiles_list, add_hs=True, kekulize=True):
    mols = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            mols.append(None)
            continue

        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except:
                mols.append(None)
                continue

        if add_hs:
            mol = Chem.AddHs(mol)

        mols.append(mol)

    return mols

def mord_descriptors(mol):
    eta_calc = Calculator([ExtendedTopochemicalAtom, AcidBase, Aromatic, AtomCount, BondCount,
                           CarbonTypes, Constitutional, EState, HydrogenBond, Lipinski, PathCount, Polarizability,
                           RingCount, RotatableBond, SLogP, TopoPSA, Weight], ignore_3D=True)
    descriptors = eta_calc(mol)
    return descriptors.asdict() 

def descriptor_calculator(smiles_list:list, type:str, rad=1):
    if type=="fcfp":
        fp_list=[]
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fp_list.append(np.zeros(2048, dtype=int))
                continue
            
            fp = AllChem.GetMorganFingerprint(
                        mol,
                        radius=rad,
                        useCounts=True,
                        useFeatures=True)
            arr = np.zeros(2048, dtype=int)
            for k, v in fp.GetNonzeroElements().items():
                arr[k % 2048] += v 
            fp_list.append(arr)
        df = pd.DataFrame(
            fp_list,
            columns=[f"FCFP_{i}" for i in range(2048)])
        return df
    if type == "mordred":
        smiles1 = process_smiles(smiles_list=smiles_list)
        calc_desc = []
        for mol in smiles1:
            if mol is None:
                calc_desc.append(None)
                continue

            des = mord_descriptors(mol)
            calc_desc.append(des)
        df = pd.DataFrame(calc_desc)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df["Lipinski"] = df["Lipinski"].astype(int)
        df["GhoseFilter"] = df["GhoseFilter"].astype(int)
        return df

def standardizer(df1, df2):
    avg = df1.mean()
    stdev = df1.std()
    return (df1-avg)/stdev, (df2-avg)/stdev


hia_model = load_joblib_from_url("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/hia.joblib")
hob50_model = load_joblib_from_url("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/hob_50.joblib")
hob20_model = load_joblib_from_url("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/hob_20.joblib")
hia_tr = pd.read_excel("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/hia.xlsx", index_col=0)
hob50_tr = pd.read_excel("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/hob_50.xlsx", index_col=0)
hob20_tr = pd.read_excel("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/hob_20.xlsx", index_col=0)
hiaxtr = hia_tr.iloc[:,:-1]
hob50xtr = hob50_tr.iloc[:,:-1]
hob20xtr = hob20_tr.iloc[:,:-1]

hiaytr = hia_tr.iloc[:,-1]
hob50ytr = hob50_tr.iloc[:,-1]
hob20ytr = hob20_tr.iloc[:,-1]


des_list = pd.read_excel("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/des_list.xlsx")
hia_des = des_list["HIA"].dropna().tolist()
hob50_des = des_list["HOB_50"].dropna().tolist()
hob20_des = des_list["HOB_20"].dropna().tolist()


SECTIONS = [
        ("What is HIA?",
         "Human Intestinal Absorption (HIA) measures the fraction of an orally administered "
         "drug that is absorbed through the intestinal wall into the bloodstream. "
         "This tool return HIA class: High absorption (Class 1) or Low absorption (Class 0).\n\n"
         "  • High absorption  →  HIA ≥ 30 %\n"
         "  • Low absorption   →  HIA < 30 %\n\n"
         "High HIA is generally desirable for orally administered therapeutics."),
        ("What is HOB?",
         "Human Oral Bioavailability (HOB) reflects the fraction of an administered dose "
         "that reaches systemic circulation unchanged. It accounts for both absorption "
         "and first-pass metabolism.\n\n"
         "This tool return HOB class at two experimental cutoffs: 50% and 20%\n"
         "50% Cutoff:\n"
         "  • High bioavailability  →  HOB ≥ 50%\n"
         "  • Low bioavailability   →  HOB < 50%\n\n"
         "20% Cutoff:\n"
         "  • High bioavailability  →  HOB ≥ 20%\n"
         "  • Low bioavailability   →  HOB < 20%"),
        ("Single Mode — Input",
         "Enter a valid SMILES string in the entry box or draw the structure . Example:\n\n"
         "   CC(=O)Oc1ccccc1C(=O)O\n\n"
         "Press Run Prediction to see the 2D structure image alongside HIA and HOB predicted classes."),
        ("Single Mode — Results",
         "After prediction users will see:\n\n"
         "  2D molecular structure of the query molecule.\n"
         "  HIA and HOB predicted classes and Applicability Domain status."),
        ("Batch Mode — Input",
         "Prepare an Excel (.xlsx) file with a column header named exactly 'SMILES'. "
         "Each subsequent row should contain one SMILES string.\n"
         "The first column of the Input file must be ID column. \n\n"
         "Click Browse files to select the file, then  Run Batch Prediction. All molecules are "
         "processed in a background; results populate the table automatically."),
        ("Batch Mode — Results",
         "After prediction, the predicticed claesses will appear in the Table and "
         "Applicability Domain status will represented along with that.\n"
         "Data can be exported in CSV file format."
         "Columns:  SMILES · HIA Class · HIA AD · HOB Class (50%) · HOB AD (50%) · HOB Class (20%) · HOB AD (20%)"),
        ("Models",
         "The current version of the tool uses machine learning models to generate the predictions, using FCFP fingerprints "
         "and mordred descritors. \n"
         "HIA : Support Vector Classifier {C = 1.0, kernel = 'rbf', gamma = 'auto'} [count-based FCFP2 fingerprint]\n"
         "HOB 50% : Random Forest  {criteria = 'gini', min_samples_split = 2, min_samples_leaf = 1, depth = None, n_estimators =100}"
         " [Mordred descriptors]\n"
         "HOB 20% : Support Vector Classifier {C = 10.0, kernel = 'rbf', gamma = 'scale'} [count-based FCFP4 fingerprint]\n\n"
         "⚠️  All predictions are for research use only and must not be used for "
         "clinical decision-making without expert validation."),
    ]

other_info = [
            ("University",    "Jadavpur University"),
            ("Department",    "Department of Pharmaceutical Technology"),
            ("Address",       "Jadavpur University, Kolkata 700 032, India"),
            ("Website",       "https://sites.google.com/jadavpuruniversity.in/dtc-lab-software/home"),
        ]

def predict_molecule(smiles: str) -> dict:
    """Return placeholder HIA / HOB predictions for a SMILES string."""
    if not smiles.strip():
        raise ValueError("SMILES string is empty.")
    hia_ext_des = descriptor_calculator([smiles.strip()], type='fcfp')
    hob20_ext_des = descriptor_calculator([smiles.strip()], type='fcfp', rad=2)
    hob50_ext_des = descriptor_calculator([smiles.strip()], type='mordred')
    hia_ext_des1 = hia_ext_des[hia_des].copy()
    hob50_ext_des1 = hob50_ext_des[hob50_des].copy()
    hob20_ext_des1 = hob20_ext_des[hob20_des].copy()

    hia_lev = leverage_calculator(hiaxtr, hia_ext_des1)
    hob50_lev = leverage_calculator(hob50xtr, hob50_ext_des1)
    hob20_lev = leverage_calculator(hob20xtr, hob20_ext_des1)

    __, std_hia_ext_des1 = standardizer(hiaxtr, hia_ext_des1)
    __, std_hob50_ext_des1 = standardizer(hob50xtr, hob50_ext_des1)
    __, std_hob20_ext_des1 = standardizer(hob20xtr, hob20_ext_des1)
    
    hia_pred = hia_model.predict(std_hia_ext_des1)[0]
    hob50_pred = hob50_model.predict(std_hob50_ext_des1)[0]
    hob20_pred = hob20_model.predict(std_hob20_ext_des1)[0]


    return {
        "SMILES":    smiles.strip(),
        "HIA Class": "High" if hia_pred==1 else "Low",
        "HIA AD": hia_lev["AD Status"].iloc[0],
        "HOB Class (50%)": "High" if hob50_pred==1 else "Low",
        "HOB AD (50%)": hob50_lev["AD Status"].iloc[0],
        "HOB Class (20%)": "High" if hob20_pred==1 else "Low",
        "HOB AD (20%)": hob20_lev["AD Status"].iloc[0]
    }


def load_file(uploaded_data):
    if uploaded_data is not None:
        return pd.read_excel(uploaded_data, index_col=0)
    return None


st.set_page_config(layout="wide", page_title="OralAbsPredict", page_icon="https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/dtc.ico")


if "page" not in st.session_state:
    st.session_state.page = "prediction"

# Sidebar
with st.sidebar:
    st.header("Predict Oral Absorption")
    st.space(3)

    if st.button("📊 Perform Prediction", use_container_width=True):
        st.session_state.page = "prediction"

    if st.button("📚 Documentation", use_container_width=True):
        st.session_state.page = "docs"

    if st.button("📌 Contact & Support", use_container_width=True):
        st.session_state.page = "contact"



if st.session_state.page == "home":
    st.markdown("### Welcome to OralAbsPredict")
    st.write("Select an option from the sidebar.")

elif st.session_state.page == "prediction":
    st.markdown("## 📊 Run Prediction")
    st.write(" Predict Human Intestinal Absorption (HIA) and Human Oral Bioavailability (HOB) for Molecules.")
    tab1, tab2 = st.tabs(["Single Mode", "Batch Mode"])
    with tab1:
        st.write("Predict HIA & HOB for a single molecule using its SMILES string.")
        col1, col2  = st.columns(2)
        with col1:
            smiles = streamlit_ketcher.st_ketcher()
            input_smiles = st.text_input("Or paste SMILES string here:", value=smiles, placeholder="Enter SMILES string")
            st.space(16)
            
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol2:
                def display():
                    with col2:
                        if not input_smiles.strip():
                            st.write("Please enter a valid SMILES string.")
                        else:
                            img = smiles_to_pil(input_smiles)
                            st.image(img)
                            results=predict_molecule(input_smiles)
                            st.markdown("#### Prediction Results:")
                            st.write(f"**HIA Class:** {results['HIA Class']} Absoption [{'HIA ≥ 30%' if results['HIA Class'] =='High' else 'HIA < 30%'}] (AD: {results['HIA AD']})")
                            st.write(f"**HOB Class (50%):** {results['HOB Class (50%)']} Bioavailability [{'HOB ≥ 50%' if results['HOB Class (50%)'] =='High' else 'HOB < 50%'}] (AD: {results['HOB AD (50%)']})")
                            st.write(f"**HOB Class (20%):** {results['HOB Class (20%)']} Bioavailability [{'HOB ≥ 20%' if results['HOB Class (20%)'] =='High' else 'HOB < 20%'}] (AD: {results['HOB AD (20%)']})")
                main_but = st.button("Run Prediction", on_click=display)
        with col2:
            subcol_1, subcol_2, subcol_3 = st.columns(3)
            
            
           
            
    
    with tab2:
        st.write("Predict HIA & HOB for multiple molecules by uploading a .xlsx file with SMILES strings.")
        
        exp_data = pd.read_excel("https://raw.githubusercontent.com/004Souvik/OralAbsPredict/main/lib/sample_file.xlsx")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            exp_data.to_excel(writer, index=False)

        buffer.seek(0)
        exp_but = st.download_button(   label="Sample File",
                                        data=buffer,
                                        file_name="sample_file.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        target_file = st.file_uploader("", type=["xlsx"])
        target_df = load_file(target_file)
        
        def main_batch():
            results1 = []
            for i in range(len(target_df)):
                res = predict_molecule(target_df["SMILES"].iloc[i])
                results1.append(res)
            results_df = pd.DataFrame(results1, index=target_df.index)
            with tab2:
                st.markdown("#### Batch Prediction Results:")
                st.dataframe(results_df)

        main_but1 = st.button("Run Batch Prediction", on_click=main_batch)


elif st.session_state.page == "docs":
    st.markdown("## 📚 Documentation")
    st.write("Reference guide for the OralAbsPredict application.")
    for title, content in SECTIONS:
        with st.expander(f"📖 {title}", expanded=False):
            st.markdown(content)
    


elif st.session_state.page == "contact":
    st.markdown("## 📌 Contact & Support")
    st.write("Get in touch with the development team.")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### 👩‍🔬 Principal Investigator")
        st.write("Prof. Kunal Roy")
        st.write("Drug Theoretics and Cheminformatics (DTC) Laboratory")
        st.write("Department of Pharmaceutical Technology")
        st.write("Email: kunal.roy@jadavpuruniversity.in")

    with col_2:
        st.markdown("### 💻 Software Developer")
        st.write("Souvik Pore")
        st.write("Drug Theoretics and Cheminformatics (DTC) Laboratory")
        st.write("Department of Pharmaceutical Technology")
        st.write("Email: souvikpore123@gmail.com")
    
    for title1, content1 in other_info:
        with st.expander(f"{title1}", expanded=True):
            st.markdown(content1)

    st.space(26)

    hide_footer_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        <div class="footer">
        <p>Found an issue? email at souvikpore123@gmail.com with subject 'OralAbsPredict Bug'.</p>
        </div>
        """
    st.markdown(hide_footer_style, unsafe_allow_html=True)
