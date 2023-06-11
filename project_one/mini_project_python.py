import pandas as pd
from operator import attrgetter
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

id_p=pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-karamnov-39/1.Промежуточнй проект/olist_customers_dataset.csv')
orders=pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-karamnov-39/1.Промежуточнй проект/olist_orders_dataset.csv', parse_dates=['order_purchase_timestamp', 'order_estimated_delivery_date', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'])
olist_order_items=pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-i-karamnov-39/1.Промежуточнй проект/olist_order_items_dataset.csv', parse_dates=['shipping_limit_date'])

# Нам нужно посчитать все заказы которые выполнены или уже выполняются после подтверждения, те. прошла оплата.
# processing — в процессе сборки заказа, значит что пользователь подтвердил заказ и начали его сборку.
# shipped — отгружен со склада уже идёт к польльзователю.
# delivered — доставлен пользователю это значит заказ уже выполнет. Затем уже можно посчитать оставшихся пользователей которые заказали только один тавар

id_p_orders = id_p.merge(orders)
buy =  id_p_orders.query("order_status == ['processing', 'shipped', 'delivered']") \
            .groupby("customer_unique_id", as_index=False) \
            .agg({"customer_zip_code_prefix": "count"}) \
            .query("customer_zip_code_prefix == 1").shape[0]

# 91814 столько пользователей совершили покупку один раз

# Проверим какие есть виды статусов

orders[orders.order_delivered_customer_date.isna()].order_status.unique()

# Поскольку только delivered является доставленым заказом значит мы уберём только его и оставим всё остальное не являющимся заказом доставленым
# Создадим и перезапишем ДФ с не доставлеными заказами

no_delivered = orders[orders.order_delivered_customer_date.isna()].query("order_status != 'delivered'")

# Поскольку нам нужны данные за месяц приведём данные к месяцам

no_delivered['week']=no_delivered['order_estimated_delivery_date'].dt.strftime('%Y-%m')
no_delivered \
    .groupby(["week", "order_status"], as_index=False) \
    .agg({"customer_id": "count"}) \
    .groupby("order_status") \
    .agg({"customer_id": "mean"}) \
    .round() \
    .sort_values("customer_id")

# Вот мы и пололучили среднее значение для каждой причины

# Нам нужены данные с покупками и датами

product = orders.merge(olist_order_items)
product.head()

# Оставляем статусы только с покупками

proba_bye = product.query("order_status == ['processing', 'shipped', 'delivered']")
proba_bye.order_purchase_timestamp = proba_bye.order_purchase_timestamp.dt.day_name()
proba_bye .groupby(['product_id', "order_purchase_timestamp"], as_index=False) \
            .agg({"customer_id": "count"}) \
            .sort_values(by="customer_id", ascending=False) \
            .drop_duplicates(subset='product_id')

# Cоберём все датафреймы в один и приведём данные к месяцам

all_ = id_p.merge(orders)
all_df = all_.merge(olist_order_items)
all_df.order_purchase_timestamp = all_df.order_purchase_timestamp.dt.strftime('%Y-%m')
all_df.head(1)

# Создадим колонку с месяцем покупки и количеством недель в месяце

all_df['delivered_month']=all_df['order_delivered_customer_date'].dt.strftime('%B')
#месяц
all_df['week_in_month']=all_df.order_delivered_customer_date.dt.daysinmonth/7
#неделя месяца

# Посчитаем кол-во покупок по каждому покупателю и каждому заказу для пользователя

purchase_month=all_df.groupby(['customer_unique_id', 'order_id', 'delivered_month', 'week_in_month'], as_index=False) \
                            .agg({'order_item_id':'count'})  \
                            .rename(columns={'order_item_id':'count_purchase'})
purchase_df_month.head(5)

# Подсчитаем среднее колличество покупок в неделю

otvet=purchase_month.groupby(['customer_unique_id', 'delivered_month'], as_index=False)  \
                    .agg({'count_in_week':'mean'})
otvet.head(5)

df5 =id_p.merge(orders)

# Для когортного анализа нам нужны:
# 1) первый месяц свершения заказа;
# 2) история совершения заказов по месяцам;
# 3) количество уникальных пользователей, которые совершили заказ в тот или иной месяц.
# Переведём колонку в нужный формат

df5['order_purchase_timestamp']=pd\
    .to_datetime(df5['order_purchase_timestamp'], format='%Y-%m-%d')

# Из даты заказа достану месяц и добавлю в новый столбец

df5['month_purchase'] = df5['order_purchase_timestamp'].dt.to_period('M')

# Беру минимальный месяц первого заказа

df5['cohort'] = df5.groupby('customer_unique_id')['order_purchase_timestamp'] \
                 .transform('min') \
                 .dt.to_period('M')

# По когорте и месяцу закупки считаем количество уникальных пользователей
# и фильтруем данные с января по декабрь

df5 = df5.groupby(['cohort', 'month_purchase']) \
              .agg(count_customer=('customer_unique_id', 'nunique')) \
              .reset_index(drop=False)
df5 = df5.query('(cohort>="2017-01") and (cohort<="2017-12")')

# Определяю номер месятца в который была совершена покупка

df5['month_purchase_number'] = df5.month_purchase.astype('int') - df5.cohort.astype('int')

# Для удобства делаю таблицу

cohort = df5.pivot(index='cohort', columns='month_purchase_number', values='count_customer')
cohort = cohort.div(cohort[0], axis=0)

# Когорта за 3 месяца

cohort_3_month = cohort[3].sort_values(ascending=False).head(1) * 100
cohort_3_month

# Построю heatmap для того чтобы было ясней в какой период идёт хорошое удержание клиентов
# чем график светлей тем лучше для бинеса здачит идёт доход
# чем темней тем наоборот хуже

fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(cohort*100, annot=True, ax=ax, linewidths=.3, fmt='.2f', vmax=0.7)

df_s=id_p.merge(orders)
df_6=df_s.merge(olist_order_items)

# Проверим есть ли пустые ID

print('{:,} без наименования ID'.format(df_6[df_6.customer_id.isnull()].shape[0]))

# Временные рамки таблицы

print('Начало {} конец {}'.format(df_6['order_approved_at'].min(),
                                    df_6['order_approved_at'].max()))
df_6 = df_6.query('order_status != ["unavailable", "created", "invoiced", "canceled"]')

# Создадим колонку с годом когда он оплачен

df_6['year'] = df_6.order_approved_at.dt.year
df_6.head(1)

# Так как для расчёта CRR обычно используют целый год возьмём 2017 г. поскольку он у нас полный

df_6 = df_6.query ('year == "2017"')

# Добавим дату покупки +1 чтобы условно у нас получилась сегодняшняя дата

today = df_6 ['order_approved_at'].max() + timedelta(days=1)
today

# время от последней покупки пользователя до текущей даты

r = df_6.groupby('customer_unique_id')\
    .agg ({'order_approved_at': 'max'})
r.head()

# Добавим столбец с количеством дней между покупкой и настоящим моментом

r['days_in_order'] = r['order_approved_at'].apply(lambda x: (today - x).days)
r.head()

# Посчитаем частоту покупок клиентами

f = df_6.groupby ('customer_unique_id')\
            .agg ({'order_id': 'count'}).reset_index()
f.head()

# Расчитаем сумму по покупкам для каждого пользователя за всё время

m = df_6.groupby(['customer_unique_id']).agg({'price': lambda x: x.sum()}).reset_index()
m.head()

# Oбеденим таблицы

rf = r.merge(f, on = 'customer_unique_id')
RFM = rf.merge(m, on = 'customer_unique_id')
RFM.head()

# Используем квантили для разбивки

quintiles = RFM[['days_in_order', 'order_id', 'price']].quantile([.2, .4, .6, .8]).to_dict()
quintiles

# Теперь пишу функцию для присвоения раногов от 1-5.Чем меньше значение days_in_order, тем лучше. Чем больше остальные показатели тем лучше

def r_score(x):
    if x <= quintiles['days_in_order'][.2]:
        return 5
    elif x <= quintiles['days_in_order'][.4]:
        return 4
    elif x <= quintiles['days_in_order'][.6]:
        return 3
    elif x <= quintiles['days_in_order'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5

# Присвоим рейтинги

RFM['R'] = RFM['days_in_order'].apply(lambda x: r_score(x))
RFM['F'] = RFM['order_id'].apply(lambda x: fm_score(x, 'order_id'))
RFM['M'] = RFM['price'].apply(lambda x: fm_score(x, 'price'))

# Создадим стобец с оценками

RFM['Score'] = RFM['R'].map(str) + RFM['F'].map(str) + RFM['M'].map(str)
RFM

# Клиенты по RFM:
# 555 — Лояльные и активные: покупали недавно, покупают часто и много платят;
# 111 — Отток: покупали давно, покупают редко и мало платят;
# R=[1, 2], F=[4, 5], M=[4, 5] — потерявшие активность: покупали давно, но часто и много платили