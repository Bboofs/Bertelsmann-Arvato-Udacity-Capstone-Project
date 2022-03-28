# using python 3

import pickle
import pandas as pd
from flask import Flask
from flask import url_for
from flask import redirect
from flask import render_template
from random import randrange
from wtforms import RadioField
from wtforms import SelectField
from wtforms import StringField
from wtforms import SubmitField
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
# from web_app.cols import AZDIAS_COLS
from web_app.columns_and_values import COLS_AND_VALS


app = Flask(__name__)
# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'some?bamboozle#string-foobar'
# Flask-Bootstrap requires this line
Bootstrap(app)
# this turns file-serving to static, using Bootstrap files installed in env
# instead of using a CDN
app.config['BOOTSTRAP_SERVE_LOCAL'] = True


def get_form_options():
    # load data descriptions
    attributes_values = pd.read_excel('../data/DIAS Attributes - Values 2017.xlsx', skiprows=1, usecols=['Attribute', 'Description', 'Value', 'Meaning'])
    attributes_info = pd.read_excel('../data/DIAS Information Levels - Attributes 2017.xlsx', skiprows=1, usecols=['Information level', 'Attribute', 'Description', 'Additional notes'])
    # print(attributes_values.info)

    # forward fill all the rowname values in the atributes table
    attributes_values['Attribute'] = attributes_values['Attribute'].ffill()

    cnt = 1
    options = ''
    for col_name, values_ in COLS_AND_VALS.items():

        for_ = name_ = id_ = col_name

        small_df = attributes_values[attributes_values['Attribute'] == col_name][['Value', 'Meaning']]

        # values_ = list(azdias_[col_name].value_counts().sort_index().index)
        all_items = len(values_)
        selected = randrange(all_items)

        try:
            label = attributes_info[attributes_info['Attribute'] == col_name]['Description'].values[0]
        except:
            label = name_

        if all_items <= 5:
            options = options + '\n' + col_name + ' = RadioField("' + str(cnt) + '. ' + col_name + '", choices=['
        else:
            options = options + '\n' + col_name + ' = SelectField("' + str(cnt) + '. ' + label + '", choices=['

        for val in values_:
            try:
                meaning = small_df[small_df['Value'] == val]['Meaning'].values[0]
            except:
                meaning = val
            if all_items <= 5:
                options = options + '(' + str(val) + ', "' + str(meaning) + '"), '
            else:
                options = options + '(' + str(val) + ', "' + str(meaning) + '"), '

        options = options.rstrip(', ')
        options = options + '], default="' + str(values_[selected]) + '")'

        cnt += 1

        #     print(col_name)
        #     print('*'*50)
    # print(options)
    return options


# with Flask-WTF, each web form is represented by a class
# "CustomerProfileForm" can change; "(FlaskForm)" cannot
# see the route for "/" and "index.html" to see how this is used
class CustomerProfileForm(FlaskForm):
    submit1 = SubmitField('Submit')

    exec(get_form_options())

    submit2 = SubmitField('Submit')


# all Flask routes below

# two decorators using the same function
@app.route('/', methods=['GET', 'POST'])
@app.route('/index.html', methods=['GET', 'POST'])
def index():
    # you must tell the variable 'form' what you named the class, above
    # 'form' is the variable name used in this template: index.html

    form = CustomerProfileForm()
    # form = None
    # if form is None:
    #     form = CustomerProfileForm()

    message = ""

    lnr = randrange(1000000, 9999999)
    cols = []
    rows = []

    if form.validate_on_submit():
        for field in form:
            # these are available fields to the form:
            # print('field.name:', field.name)
            # print('field.description:', field.description)
            # print('field.label.text:', field.label.text)
            # print('field.data:', field.data)
            # print('*'*30)

            if field.name not in ['submit', 'submit1', 'submit2', 'csrf_token']:
                cols.append(field.name)
                rows.append(float(field.data))

        # convert the submitted data into a df
        row_df = pd.DataFrame([rows], columns=cols)
        # print(row_df.info())

        # load fitted model from file --xgb
        best_xgb_ = pickle.load(open("../data/fitted.best_xgb.pickle.dat", "rb"))
        preds_xgb_ = best_xgb_.predict_proba(row_df)[:, 1]
        kaggle_xgb_ = pd.DataFrame(index=[lnr], data=preds_xgb_)
        kaggle_xgb_.rename(columns={0: "RESPONSE"}, inplace=True)
        prob_xgb_ = kaggle_xgb_['RESPONSE'].values[0]
        # print(kaggle_xgb_.head())
        # print(prob_xgb_)

        # load fitted model from file --adaboost
        best_adaboost_ = pickle.load(open("../data/fitted.best_adaboost.pickle.dat", "rb"))
        preds_adaboost_ = best_adaboost_.predict_proba(row_df)[:, 1]
        kaggle_adaboost_ = pd.DataFrame(index=[lnr], data=preds_adaboost_)
        kaggle_adaboost_.rename(columns={0: "RESPONSE"}, inplace=True)
        prob_adaboost_ = kaggle_adaboost_['RESPONSE'].values[0]
        # print(kaggle_adaboost_.head())
        # print(prob_adaboost_)

        # form = None

        return redirect(url_for('customer', lnr=lnr, prob_xgb_=prob_xgb_, prob_adaboost_=prob_adaboost_))
    return render_template('index.html', form=form, message=message)


@app.route('/customer/<lnr>/<prob_xgb_>/<prob_adaboost_>')
def customer(lnr, prob_xgb_, prob_adaboost_):
    # pass all the data for the selected actor to the template
    return render_template('customer.html', lnr=lnr, prob_xgb=prob_xgb_, prob_adaboost=prob_adaboost_)


# routes to handle errors - they have templates too

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


# keep this as is
if __name__ == '__main__':
    app.run(debug=True, port=4321)
