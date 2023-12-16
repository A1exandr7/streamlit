from scipy.stats import ttest_ind
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import os


plt.style.use('ggplot')
sns.set_style('darkgrid')
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('font', size= 12)

st.title('Анализ заболеваний сотрудников')



file = st.file_uploader("Загрузите данные", type="csv")
if file is not None:
    data = pd.read_csv(file, encoding='cp1251',  quotechar=',')
    st.write("Данные успешно загружены")
else:
    file = os.path.join(os.getcwd(), 'М.Тех_Данные_к_ТЗ_DS.csv')
    data = pd.read_csv(file, encoding='cp1251',  quotechar=',')

#предобработка данных
data.rename(columns={'"Количество больничных дней': 'Количество больничных дней',
                     '""Возраст""': 'Возраст',
                     '""Пол"""': 'Пол'}, inplace=True)
data['Количество больничных дней'] = data['Количество больничных дней'].str.strip('"').astype(int)
data['Пол'] = data['Пол'].str.strip('"')
st.write(data.head())

st.sidebar.title("Знакомство с данными")
#знакомство с данными
st.sidebar.header("Статистика")
if st.sidebar.button("Информация о таблице"):
    st.sidebar.write(f"Количество сотрудников: {len(data)}")
    st.sidebar.write("Количество пропусков в данных")
    missing_values = data.isna().sum().to_frame(name='Количество').rename_axis('Признак')
    st.sidebar.dataframe(missing_values)
    st.sidebar.write("Описательная статистика:")
    st.sidebar.write(data.describe())

if st.sidebar.button("Статистика по сотрудникам(М&Ж)"):
    st.sidebar.write("Количество Мужчин и Женщин:")
    male_female_count = data['Пол'].value_counts().to_frame(name='Количество')
    st.sidebar.dataframe(male_female_count)

    st.sidebar.text("")
    st.sidebar.write("Средний возраст сотрудников:")
    age_employees = data.groupby('Пол')['Возраст'].agg(['mean'])
    st.sidebar.dataframe(age_employees)

    st.sidebar.text("")
    st.sidebar.write("Средняя продолжительность больничного:")
    sick_days_avgerage = data.groupby('Пол')['Количество больничных дней'].agg(['mean'])
    st.sidebar.dataframe(sick_days_avgerage)


st.sidebar.header("Графики")
if st.sidebar.button("Распределение возраста"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=data, x='Возраст', hue='Пол', bins=np.arange(data['Возраст'].min(),
                                                                   data['Возраст'].max() + 1, 1))
    plt.title('Распределение Возраста')
    plt.ylabel('Кол-во сотрудников')
    st.pyplot(fig)


if st.sidebar.button("Распределение больничных дней"):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=data,x='Количество больничных дней', hue='Пол', bins=20)
    plt.title('Распределение больничных дней')
    plt.ylabel('Кол-во сотрудников')
    st.pyplot(fig)

st.header("Проверка гипотез")
days = st.selectbox("Выберите Количество дней на больничном", np.unique(data['Количество больничных дней']))
age = st.selectbox("Выберите возраст", np.unique(data['Возраст']))
alpha = st.slider("Выберите уровень значимости alpha:", 0.0, 1.0, 0.5, step=0.01)
st.write("Выбрано дней:", days)
st.write("Выбран уровень значимости:", alpha)



def get_student_parameter(data_one: np.array, data_two: np.array) -> float:

    """Вычисляет параметр функции распределения Стьюдента."""
    len_one, len_two = len(data_one), len(data_one)
    mean_one, mean_two = np.mean(data_one), np.mean(data_two)
    std_one, std_two = np.std(data_one), np.std(data_two)
    k = (
        ((std_one ** 2) / len_one + (std_two ** 2) / len_two) ** 2
        / (
            (std_one ** 4) / ((len_one ** 2) * (len_one - 1))
            + (std_two ** 4) / ((len_two ** 2) * (len_two - 1))))
    return k

#гипотеза 1
if st.button(f"Проверить гипотезу: Мужчины пропускают в течение года \nболее {days} рабочих дней по болезни значимо чаще женщин."):
    data[f'Больничный>{days}дней'] = (data['Количество больничных дней']>days)*1
    percent_emp = round(data[f'Больничный>{days}дней'].mean()*100, 2)
    st.write(f'Процент сотрудников уходящих на больничный более {days} дней: {percent_emp}%')
    male_female_percent = data.groupby('Пол')[f'Больничный>{days}дней'].agg(['mean', 'count'])
    male_female_percent['Процент'] = male_female_percent['mean'] * 100
    male_female_percent['Количество'] = male_female_percent['count']
    male_female_percent.drop(['mean', 'count'], axis=1, inplace=True)
    st.text("")
    st.write(f'При условии что данные - все сотрудники компании, решение о значимости пропусков можно принять по данной таблице')
    st.dataframe(male_female_percent)
    st.write(f'При условии что данные - выборка:')
    st.text("")
    a = data[data['Пол']=='М'][f'Больничный>{days}дней']
    b = data[data['Пол']=='Ж'][f'Больничный>{days}дней']
    t_test = ttest_ind(a, b, equal_var=False)
    results_df = pd.DataFrame({'Название теста': ['t-test'],
                               'pvalue': [t_test.pvalue],
                               'statistic': [t_test.statistic]})

    st.dataframe(results_df)
    if t_test.pvalue<alpha:
        st.write('Статистически значимые отличия')
    else:
        st.write('Отличий между группами нет')

    k = get_student_parameter(a, b)
    critical_region = stats.t.ppf([alpha/2, 1 - alpha/2], df=k)
    X = np.linspace(-3, 3, 1000)
    Y = stats.t.pdf(X, k)
    critical_mask_list = [X < critical_region[0], X > critical_region[1]]


    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, mask in enumerate(critical_mask_list):
        X_ = X[mask]
        Y_upper = Y[mask]
        Y_down = np.zeros(len(Y_upper))
        plt.fill_between(X_, Y_down, Y_upper,
                         color='r', alpha=0.3,
                         label='critical region' if idx==0 else ''
                        )
    plt.plot(X, Y, label=f'St (k={k:0.0f})')
    plt.scatter([t_test.statistic], [0], color='k', label='t-statistic')
    plt.legend()
    st.pyplot(fig)
    st.write('Данный график интерпритируется следующим образом:')
    st.write('Если t-статистика попала в критические области(отмечены красным), то следует отклонить гипотезу о отсутствии различий')
    st.write('Если t-статистика не попала в критические области, отличий между группами нет')

#гипотеза 2
if st.button(f"Проверить гипотезу: Сотрудники старше {age} пропускают больше {days} чем молодые коллеги"):
    data[f'Больничный>{days}дней'] = (data['Количество больничных дней']>days)*1
    data[f'Возраст>{age}'] = (data['Возраст']>age)*1
    old_young_percent = data.groupby(f'Возраст>{age}')[f'Больничный>{days}дней'].agg(['mean', 'count'])
    old_young_percent['Процент'] = old_young_percent['mean'] * 100
    old_young_percent['Количество'] = old_young_percent['count']
    old_young_percent.drop(['mean', 'count'], axis=1, inplace=True)
    st.text("")
    st.write(f'При условии что данные - все сотрудники компании, решение о значимости пропусков можно принять по данной таблице')
    st.dataframe(old_young_percent)
    st.write(f'При условии что данные - выборка:')
    st.text("")
    a = data[data['Возраст'] > age][f'Больничный>{days}дней']
    b = data[data['Возраст'] < age][f'Больничный>{days}дней']
    st.text("")
    t_test = ttest_ind(a, b, equal_var=False)
    results_df = pd.DataFrame({'Название теста': ['t-test'],
                               'pvalue': [t_test.pvalue],
                               'statistic': [t_test.statistic]})

    st.dataframe(results_df)
    if t_test.pvalue<alpha:
        st.write('Статистически значимые отличия')
    else:
        st.write('Отличий между группами нет')

    k = get_student_parameter(a, b)
    critical_region = stats.t.ppf([alpha/2, 1 - alpha/2], df=k)
    X = np.linspace(-3, 3, 1000)
    Y = stats.t.pdf(X, k)
    critical_mask_list = [X < critical_region[0], X > critical_region[1]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, mask in enumerate(critical_mask_list):
        X_ = X[mask]
        Y_upper = Y[mask]
        Y_down = np.zeros(len(Y_upper))
        plt.fill_between(X_, Y_down, Y_upper,
                         color='r', alpha=0.3,
                         label='critical region' if idx==0 else ''
                        )
    plt.plot(X, Y, label=f'St (k={k:0.0f})')
    plt.scatter([t_test.statistic], [0], color='k', label='t-statistic')
    plt.legend()
    st.pyplot(fig)
    st.write('Данный график интерпритируется следующим образом:')
    st.write('Если t-статистика попала в критические области(отмечены красным), то следует отклонить гипотезу о отсутствии различий')
    st.write('Если t-статистика не попала в критические области, отличий между группами нет')
