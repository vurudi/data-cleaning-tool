# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'raw_data'
CLEANED_FOLDER = 'cleaned_data'  # Add a folder to save cleaned data
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLEANED_FOLDER'] = CLEANED_FOLDER

ALLOWED_EXTENSIONS = {'json', 'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dataset_overview(file_path):
    try:
        # Read the file into a DataFrame based on file extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Get dataset overview
        num_records = len(df)
        num_columns = len(df.columns)
        column_names = df.columns.tolist()
        data_types = df.dtypes
        
        overview = {
            'num_records': num_records,
            'num_columns': num_columns,
            'column_names': column_names,
            'data_types': data_types,
            'missing_values': [],
            'num_duplicates': 0,  # New attribute for duplicates
            'incompatible_data': []
        }

        # Calculate missing values for each column
        for column in df.columns:
            num_missing = df[column].isnull().sum()
            missing_percentage = num_missing / len(df) * 100
            overview['missing_values'].append({
                'column_name': column,
                'num_missing': num_missing,
                'missing_percentage': f'{missing_percentage:.2f}%'
            })

        # Calculate number of duplicates
        overview['num_duplicates'] = df.duplicated().sum()
        
        return overview
    except Exception as e:
        return str(e)


from scipy import stats

def clean_data(file_path, missing_values_strategy=None, duplicate_columns=None, outlier_strategy=None):
    try:
        # Read the dataset
        df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

        # Drop specific columns
        columns_to_drop = ['name', 'email', 'contact']
        df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')  # Ignore error if column doesn't exist

        # Handling missing values
        if missing_values_strategy == 'imputation':
            # Impute missing values with mean for columns with less than 30% missing values, else drop the column
            for column in df.columns:
                missing_percentage = df[column].isnull().mean() * 100
                if missing_percentage <= 30:
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    df.drop(column, axis=1, inplace=True)
        elif missing_values_strategy == 'delete_rows':
            # Delete rows with missing values
            df.dropna(inplace=True)
        elif missing_values_strategy == 'delete_columns':
            # Delete columns with missing values
            df.dropna(axis=1, inplace=True)

        # Removing duplicates
        if duplicate_columns:
            df.drop_duplicates(subset=duplicate_columns, inplace=True)

        # Dealing with outliers
        if outlier_strategy == 'remove_outliers':
            # Remove outliers using z-score method
            z_scores = stats.zscore(df.select_dtypes(include='number'))
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            df = df[filtered_entries]
        elif outlier_strategy == 'transform_outliers':
            # Transform outliers using log transformation
            df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').apply(
                lambda x: np.log(x + 1) if np.issubdtype(x.dtype, np.number) else x)
        elif outlier_strategy == 'treat_separately':
            # Treat outliers separately (e.g., replace with median)
            for column in df.select_dtypes(include='number').columns:
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].median(),
                                      df[column])

        # Save the cleaned data to a new file
        cleaned_file_path = os.path.join('cleaned_data', 'cleaned_' + os.path.basename(file_path))
        df.to_csv(cleaned_file_path, index=False)
        return cleaned_file_path
    except Exception as e:
        return str(e)




import os
from flask import render_template, jsonify

@app.route('/')
def index():
    upload_dir = app.config['UPLOAD_FOLDER']
    files = os.listdir(upload_dir)
    return render_template('index.html', files=files)

@app.route('/get_files')
def get_files():
    upload_dir = app.config['UPLOAD_FOLDER']
    cleaned_dir = app.config['CLEANED_FOLDER']
    show_cleaned = request.args.get('show_cleaned')

    if show_cleaned:
        files = os.listdir(cleaned_dir)
    else:
        files = os.listdir(upload_dir)

    return jsonify(files)


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        selected_file = request.form.get('existing_file')  # Get the selected file from dropdown
        uploaded_file = request.files['file']  # Get the uploaded file from form

        if selected_file:
            cleaned_file_path = os.path.join(app.config['CLEANED_FOLDER'], selected_file)
            if os.path.exists(cleaned_file_path):
                # If the selected file exists in the cleaned folder, generate overview directly
                overview = get_dataset_overview(cleaned_file_path)
                if isinstance(overview, str):
                    flash('Error: ' + overview, 'error')
                else:
                    return render_template('overview.html', overview=overview, file_name=selected_file)
            else:
                # If the selected file does not exist in the cleaned folder, proceed with regular upload
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_file)
                overview = get_dataset_overview(file_path)
                if isinstance(overview, str):
                    flash('Error: ' + overview, 'error')
                else:
                    return render_template('overview.html', overview=overview, file_name=selected_file)
        else:
            # Proceed with regular file upload
            if uploaded_file.filename != '' and allowed_file(uploaded_file.filename):
                # Save the file to the 'raw_data' folder
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)
                
                # Get dataset overview
                overview = get_dataset_overview(file_path)
                if isinstance(overview, str):
                    flash('Error: ' + overview, 'error')
                else:
                    return render_template('overview.html', overview=overview, file_name=uploaded_file.filename)
            else:
                flash('Invalid file format. Please upload a JSON, CSV, or Excel file.', 'error')
    return redirect(url_for('index'))


@app.route('/cleaning/<file_name>', methods=['GET', 'POST'])
def cleaning(file_name):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    overview = get_dataset_overview(file_path)  # Fetch overview data
    if request.method == 'POST':
        # Handle data cleaning operations here
        flash('Data cleaning completed.', 'success')
    return render_template('cleaning.html', file_name=file_name, overview=overview)


from flask import redirect, url_for
@app.route('/save_cleaned_data', methods=['POST'])
def save_cleaned_data():
    if request.method == 'POST':
        file_name = request.form['file_name']  # Get the file name from the form
        raw_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

        # Get the selected cleaning strategies from the form
        missing_values_strategy = request.form.get('missing_values_strategy')
        duplicate_columns = request.form.getlist('duplicate_columns')
        outlier_strategy = request.form.get('outlier_strategy')

        try:
            # Read the dataset
            df = pd.read_csv(raw_file_path) if raw_file_path.endswith('.csv') else pd.read_excel(raw_file_path)

            # Drop specific columns
            columns_to_drop = ['name', 'email', 'contact']
            df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')  # Ignore error if column doesn't exist

            # Handling missing values
            if missing_values_strategy == 'imputation':
                # Impute missing values with mean for columns with less than 30% missing values, else drop the column
                for column in df.columns:
                    missing_percentage = df[column].isnull().mean() * 100
                    if missing_percentage <= 30:
                        df[column].fillna(df[column].mean(), inplace=True)
                    else:
                        df.drop(column, axis=1, inplace=True)
            elif missing_values_strategy == 'delete_rows':
                # Delete rows with missing values
                df.dropna(inplace=True)
            elif missing_values_strategy == 'delete_columns':
                # Delete columns with missing values
                df.dropna(axis=1, inplace=True)

            # Removing duplicates
            if duplicate_columns:
                df.drop_duplicates(subset=duplicate_columns, inplace=True)

            # Dealing with outliers
            if outlier_strategy == 'remove_outliers':
                # Remove outliers using z-score method
                z_scores = stats.zscore(df.select_dtypes(include='number'))
                abs_z_scores = np.abs(z_scores)
                filtered_entries = (abs_z_scores < 3).all(axis=1)
                df = df[filtered_entries]
            elif outlier_strategy == 'transform_outliers':
                # Transform outliers using log transformation
                df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').apply(
                    lambda x: np.log(x + 1) if np.issubdtype(x.dtype, np.number) else x)
            elif outlier_strategy == 'treat_separately':
                # Treat outliers separately (e.g., replace with median)
                for column in df.select_dtypes(include='number').columns:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), df[column].median(),
                                          df[column])

            # Save the cleaned data to a new file
            cleaned_file_path = os.path.join('cleaned_data', 'cleaned_' + os.path.basename(raw_file_path))
            df.to_csv(cleaned_file_path, index=False)
            
            flash('Cleaned data saved successfully!', 'success')
            
        except Exception as e:
            flash('Error saving cleaned data: ' + str(e), 'error')

    return redirect(url_for('index'))
from flask import send_file



@app.route('/download_cleaned_data/<file_name>', methods=['GET'])
def download_cleaned_data(file_name):
    cleaned_file_path = os.path.join(app.config['CLEANED_FOLDER'], file_name)
    return send_file(cleaned_file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
