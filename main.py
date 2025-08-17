import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import subprocess
import sys
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm
import os
from glob import glob

from my_tabm import apply_tabm_cv, apply_tabm_cv_tune

df_train = pd.read_csv("./dataset/train.csv")
df_test = pd.read_csv("./dataset/test.csv")
df_best = pd.read_csv('./submission_cur_best+tabm_b1234_10fold(cb)_93.92695.csv')

feature_cols = df_train.columns[:55]
df_test_pred = pd.concat([df_test, df_best], axis=1)


feature_cols = df_train.columns[:55]
target_cols = df_train.columns[55:]


if 'df_train' not in st.session_state:
    st.session_state['df_train'] = df_train

if 'df_test' not in st.session_state:
    st.session_state['df_test'] = df_test

if 'feature_cols' not in st.session_state:
    st.session_state['feature_cols'] = feature_cols

# Main page content
st.markdown("# Shell.ai Hackathon 2025")

# st.sidebar.markdown("# Main page")

# col_run_tabm, col2 = st.columns(2)

# with col_run_tabm:
df_train = st.session_state['df_train']
'X_train dimension:', df_train[feature_cols].shape
# 'y_train dimension:', df_train[target_cols].shape

models = ['tabm', 'autogluon']

for model in models:
    if model not in st.session_state:
        st.session_state[model] = {}
        st.session_state[model]['show_sidebar'] = False
        if 'seed_lower' not in st.session_state[model]:
            st.session_state[model]['seed_lower'] = 42
        if 'seed_upper' not in st.session_state[model]:
            st.session_state[model]['seed_upper'] = 42
        if 'n_trials' not in st.session_state[model]:
            st.session_state[model]['n_trials'] = 100
        if 'run_seed_lower' not in st.session_state[model]:
            st.session_state[model]['run_seed_lower'] = 42
        if 'run_seed_upper' not in st.session_state[model]:
            st.session_state[model]['run_seed_upper'] = 42
        if 'n_splits' not in st.session_state[model]:
            st.session_state[model]['n_splits'] = 5
        if 'tuner_splits' not in st.session_state[model]:
            st.session_state[model]['tuner_splits'] = 5


options = target_cols

if 'selected_target_cols' in st.session_state:
    selected_target_cols = st.session_state['selected_target_cols']

selected_target_cols = st.multiselect(
    "Select one or more target labels to predict:",
    options,
    # default=st.session_state.selected_target_cols,
)

st.session_state['selected_target_cols'] = selected_target_cols


with st.container(height=420):
    cur_model = 'tabm'
    st.header('TabM')
    # 'CV scores:'
    hparams_all = pd.read_csv("./optuna/tabm_cv/hparams_cv.csv", index_col=0)
    
    df_cv_score = pd.DataFrame()
    for target_col in selected_target_cols:
        hparams = hparams_all[hparams_all['Target'] == target_col]
        df_cv_score['BP'+target_col.split('BlendProperty')[-1]] = hparams['Score']
        best_hparams = hparams.iloc[0].to_dict()
        if f'hparams_{target_col}' not in st.session_state['tabm']:
            st.session_state['tabm'][f'hparams_{target_col}'] = best_hparams
    
    df_cv_score.index = ['CV Score']   
    st.table(df_cv_score)
    
    if len(selected_target_cols) > 0 :
        with st.expander('Hyperparameters'):
            hparams_all
        
    # if "show_sidebar_tabm" not in st.session_state:
    #     st.session_state.show_sidebar_tabm = False
        
    
        
    st.markdown('##### Run TabM')


    col_run_seed_lower, col_run_seed_lower_input, col_run_seed_upper, col_run_seed_upper_input, col_n_splits, col_n_splits_input, col_set_hyperparameters, col_run_tabm = st.columns([0.4, 0.4, .4, 0.4, 0.4, 0.4, 1.1, 0.9])
    
    with col_set_hyperparameters:
        if st.button('Set Hyperparameters', use_container_width=True):
            for model in models:
                st.session_state[model]['show_sidebar'] = False
            st.session_state[cur_model]['show_sidebar'] = True
                           
    with col_run_seed_lower:
        'Seed lower:'
    
    with col_run_seed_lower_input:
        run_seed_lower = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_run_seed_lower_input',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['run_seed_lower'] = run_seed_lower
    
    with col_run_seed_upper:
        'Seed upper:'
    
    with col_run_seed_upper_input:
        run_seed_upper = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_run_seed_upper_input',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['run_seed_upper'] = run_seed_upper
    
    
    with col_n_splits:
        'Splits:'
    
    with col_n_splits_input:
        n_splits = st.text_input(
                            label='',
                            value=5,
                            key=f'n_splits_input',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['n_splits'] = n_splits
    
    with col_run_tabm:
        if st.button('Run Tabm', use_container_width=True):
            for target_col in selected_target_cols:
                run_seed_lower = int(st.session_state['tabm']['run_seed_lower'])
                run_seed_upper = int(st.session_state['tabm']['run_seed_upper'])
                for seed in range(run_seed_lower, run_seed_upper + 1):
                    hparams = st.session_state['tabm'][f'hparams_{target_col}']
                    hparams['k'] = 32
                    df_train = st.session_state['df_train']
                    feature_cols = st.session_state['feature_cols']
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    def update_progress(current, total):
                        percent = current / total
                        progress_bar.progress(percent)
                        status_text.text(f"Processing fold {current}/{total}")
                        
                    n_splits = int(st.session_state['tabm']['n_splits'])
                    
                    score, _ = apply_tabm_cv(hparams, df_train, df_test_pred, feature_cols, target_col, seed=seed, n_splits=n_splits, callback=update_progress)
                    score
        
    
    
    col_seed_lower, col_seed_lower_input, col_seed_upper, col_seed_upper_input, col_tuner_splits, col_tuner_splits_input, col_n_trials, col_n_trials_input, col_tune_hyperparameters = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 3])
    
    with col_tune_hyperparameters:
        df_best_hparams = None
        if st.button('Tune Hyperparameters', use_container_width=True):
            
            n_trials = 100
            
            for target_col in selected_target_cols:
                def objective(trial):
                    score = apply_tabm_cv_tune(trial, df_train, df_test_pred, feature_cols, target_col, seed=100, n_splits=5)
                    return score
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def streamlit_callback(study, trial):
                    completed = len(study.trials)
                    progress_bar.progress(completed / n_trials)
                    status_text.text(f"Running trial {completed}/{n_trials}")

                study = optuna.create_study(sampler=TPESampler(), direction='maximize')
                study.optimize(objective, n_trials=n_trials, callbacks=[streamlit_callback])
                status_text.text("✅ Done!")
                st.write("Best trial:", study.best_trial.params)
                
                map_hparams = study.best_params
                map_hparams['Target'] = target_col
                map_hparams['Score'] = study.best_value
                map_hparams['Best trial'] = study.best_trial.number
                df_cur_best = pd.DataFrame([map_hparams])
                df_best_hparams = pd.concat([df_best_hparams, df_cur_best])
                os.makedirs('./optuna/tabm_cv', exist_ok=True)
                
                hparam_files = glob('./optuna/tabm_cv/*_v*')
                
                latest_run = max([int(file_name.split('_v')[-1].split('.')[0]) for file_name in hparam_files])
                
                df_best_hparams.to_csv(f'./optuna/tabm_cv/hparams_cv_v{latest_run + 1}.csv')
        
    with col_seed_lower:
        'Seed lower:'

    with col_seed_lower_input:
        
        seed_lower = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_seed_lower',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['seed_lower'] = seed_lower
    
    with col_seed_upper:
        'Seed upper:'

    with col_seed_upper_input:
        seed_upper = st.text_input(
                            label='',
                            value=42,  # default = best
                            key=f'tabm_seed_upper',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['seed_upper'] = seed_upper
    
    with col_n_trials:
        '\# of trials:'
    
    with col_n_trials_input:
        n_trials = st.text_input(
                            label='',
                            value=100,  # default = best
                            key=f'tabm_n_trials',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['n_trials'] = n_trials
    
    with col_tuner_splits:
        'Splits:' 
    
    with col_tuner_splits_input:
        tuner_splits = st.text_input(
                            label='',
                            value=5,  # default = best
                            key=f'tabm_tuner_splits',
                            label_visibility="collapsed",
                        )
        st.session_state['tabm']['tuner_splits'] = tuner_splits
    
    
    with st.sidebar:
        if st.session_state['tabm']['show_sidebar']:
            st.markdown("## Choose TabM Hyperparameters")
            for target_col in selected_target_cols:
                with st.expander(target_col):
                    best_hparams = hparams_all[hparams_all['Target'] == target_col].iloc[0].to_dict()
                    best_hparams.pop('Score')
                    best_hparams.pop('Target')
                    # st.write(best_hparams)
                    
                    # for key in best_hparams.keys():
                        # st.write(key)
                        
                        
                    # Possible categorical options
                    categorical_options = {
                        "embedding_type": ["PeriodicEmbeddings", "PiecewiseLinearEmbeddings"],
                        "arch_type": ['tabm', 'tabm-mini'],
                        "share_training_batches": ['T', 'F'],
                    }

                    # Detect categorical vs numeric
                    categorical_keys = list(categorical_options.keys())
                    numeric_keys = [k for k in best_hparams.keys() if k not in categorical_keys]

                    # st.sidebar.markdown("### Hyperparameter Tuning")

                    # Store updated params
                    updated_hparams = {}

                    # Categorical: dropdown
                    for key in categorical_keys:
                        updated_hparams[key] = st.selectbox(
                            label=key,
                            options=categorical_options[key],
                            index=categorical_options[key].index(best_hparams[key]),  # default = best
                            key=f'{target_col}_{key}_selectbox',
                        )

                    # # Numerical: text input (could use number_input too)
                    for key in numeric_keys:
                        updated_hparams[key] = st.text_input(
                            label=key,
                            value=str(best_hparams[key]),  # default = best
                            key=f'{target_col}_{key}_text_input',
                        )

                    st.write("Updated Hyperparameters:", updated_hparams)
                    
                    st.session_state['tabm'][f'hparams_{target_col}'] = updated_hparams
                    
                    
            
        
    
   
            


