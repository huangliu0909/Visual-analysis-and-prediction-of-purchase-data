import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
start_time = datetime.datetime.now()

color = sns.color_palette()
pd.options.mode.chained_assignment = None  # default='warn'

order_products_train = pd.read_csv("order_products__train.csv")
order_products_prior = pd.read_csv("order_products__prior.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")
aisles = pd.read_csv("aisles.csv")
departments = pd.read_csv("departments.csv")

plt.figure(figsize=(12, 8))
sns.countplot(x="order_dow", data=orders, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Day of week', fontsize=12)
plt.title("Frequency of order by week day", fontsize=15)
plt.show()
cnt_srs = orders.order_dow.value_counts()
print(cnt_srs)

plt.figure(figsize=(12, 8))
sns.countplot(x="order_hour_of_day", data=orders, color=color[4])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Hour of day', fontsize=12)
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()
cnt_srs = orders.order_hour_of_day.value_counts()
print(cnt_srs)

plt.figure(figsize=(12, 8))
sns.countplot(x="days_since_prior_order", data=orders, color=color[5])
plt.ylabel('Count', fontsize=12)
plt.xlabel('days_since_prior_order', fontsize=12)
plt.title("Frequency of order by days_since_prior_order", fontsize=15)
plt.show()
cnt_srs = orders.days_since_prior_order.value_counts()
print(cnt_srs)

cnt_srs = orders.eval_set.value_counts()
plt.figure(figsize=(12, 8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.ylabel('count', fontsize=12)
plt.xlabel('Eval set type', fontsize=12)
plt.title('Count of rows in each Eval set type', fontsize=15)
plt.show()
print(cnt_srs)

cnt_srs = orders.groupby("user_id")["order_number"].aggregate(np.max).reset_index()
cnt_srs = cnt_srs.order_number.value_counts()

plt.figure(figsize=(12, 8))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Maximum order number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

grouped_df = orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

plt.figure(figsize=(12, 6))
sns.heatmap(grouped_df)
plt.title("Frequency of Day of week Vs Hour of day")
plt.show()

print(order_products_train.head(5))
print("The order_products_train size is : ", order_products_train.shape)
print(order_products_prior.head(5))
print("The order_products_prior size is : ", order_products_prior.shape)
order_products_all = pd.concat([order_products_train, order_products_prior], axis=0)
print("The order_products_all size is : ", order_products_all.shape)

grouped = order_products_all.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'count'}).reset_index()
grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])
grouped = grouped.sort_values(by='Total_reorders', ascending=False)[:10]
grouped = grouped.groupby(['product_name']).sum()['Total_reorders'].sort_values(ascending=False)
print(grouped)

sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(12, 10))
plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylabel('Number of Reorders', fontsize=13)
plt.xlabel('Most ordered Products', fontsize=13)
plt.show()
grouped = order_products_all.groupby("reordered")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped['Ratios'] = grouped["Total_products"].apply(lambda x: x /grouped['Total_products'].sum())
print(grouped)

grouped = grouped.groupby(['reordered']).sum()['Total_products'].sort_values(ascending=False)
sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(5, 8))
sns.barplot(grouped.index, grouped.values, palette='RdBu_r')
plt.ylabel('Number of Products', fontsize=13)
plt.xlabel('Reordered or Not ', fontsize=13)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

grouped = order_products_all.groupby("product_id")["reordered"].aggregate({'reorder_sum': sum,'reorder_total': 'count'}).reset_index()
grouped['reorder_probability'] = grouped['reorder_sum'] / grouped['reorder_total']
grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])
grouped = grouped[grouped.reorder_total > 75].sort_values(['reorder_probability'], ascending=False)[:10]
grouped = grouped.groupby(['product_name']).sum()['reorder_probability'].sort_values(ascending=False)

sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(12, 10))
plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylim([0.85,0.95])
plt.ylabel('Reorder probability', fontsize=13)
plt.xlabel('Most reordered products', fontsize=12)
plt.show()
print(grouped)

print(products.head(5))
print("The products size is : ", products.shape)
print(aisles.head(5))
print("The aisles size is : ", products.shape)
print(departments.head(5))
print("The departments size is : ", products.shape)
items = pd.merge(left =pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
print(items.head())

grouped = items.groupby("department")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped['Ratio'] = grouped["Total_products"].apply(lambda x: x / grouped['Total_products'].sum())
grouped.sort_values(by='Total_products', ascending=False, inplace=True)
print(grouped)
grouped = grouped.groupby(['department']).sum()['Total_products'].sort_values(ascending=False)

sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(12, 15))
plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylabel('Number of products', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.show()

grouped = items.groupby(["department", "aisle"])["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped.sort_values(by='Total_products', ascending=False, inplace=True)
fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))
for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):
    g = sns.barplot(group.aisle, group.Total_products , ax=ax)
    ax.set(xlabel = "Aisles", ylabel=" Number of products")
    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)
    ax.set_title(aisle, fontsize=15)
plt.show()

# Most important Aisles over all Departments (by number of Products)
grouped = items.groupby("aisle")["product_id"].aggregate({'Total_products': 'count'}).reset_index()
grouped['Ratio'] = grouped["Total_products"].apply(lambda x: x / grouped['Total_products'].sum())
grouped = grouped.sort_values(by='Total_products', ascending=False)[:20]
grouped = grouped.groupby(['aisle']).sum()['Total_products'].sort_values(ascending=False)
print(grouped)


plt.figure(figsize=(10, 10))
labels = (np.array(grouped.index))
sizes = (np.array(grouped.values))
colors = 'lightgreen', 'gold', 'lightskyblue', 'lightcoral','lightpink','lightcyan','mediumpurple','silver'
plt.pie(sizes, labels=labels,colors=colors,
        autopct='%1.1f%%', startangle=200)
plt.show()

users_flow = orders[['user_id', 'order_id']].merge(order_products_train[['order_id', 'product_id']],
                                    how='inner', left_on='order_id', right_on='order_id')
users_flow = users_flow.merge(items, how='inner', left_on='product_id', right_on='product_id')
# Best Selling Departments (number of Orders)
grouped = users_flow.groupby("department")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped['Ratio'] = grouped["Total_orders"].apply(lambda x: x /grouped['Total_orders'].sum())
grouped.sort_values(by='Total_orders', ascending=False, inplace=True)
grouped = grouped.groupby(['department']).sum()['Total_orders'].sort_values(ascending=False)

f, ax = plt.subplots(figsize=(12, 15))
plt.xticks(rotation='vertical')
sns.barplot(grouped.index, grouped.values)
plt.ylabel('Number of Orders', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.show()
print(grouped)

# Best Selling Aisles in each Department (number of Orders)¶
grouped = users_flow.groupby(["department", "aisle"])["order_id"].aggregate({'Total_orders': 'count'}).reset_index()
grouped.sort_values(by='Total_orders', ascending=False, inplace=True)
fig, axes = plt.subplots(7, 3, figsize=(20, 45), gridspec_kw=dict(hspace=1.4))
i = 0
for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):

    g = sns.barplot(group.aisle, group.Total_orders , ax=ax)
    ax.set(xlabel = "Aisles", ylabel=" Number of Orders")
    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)
    ax.set_title(aisle, fontsize=15)


plt.show()

prior = order_products_prior[0:300000]
_mt = pd.merge(prior, products, on=['product_id', 'product_id'])
_mt = pd.merge(_mt, orders, on=['order_id', 'order_id'])
mt = pd.merge(_mt, aisles, on=['aisle_id', 'aisle_id'])
print(mt.head(10))

cust_prod = pd.crosstab(mt['user_id'], mt['aisle'])
print(cust_prod.head(10))

pca = PCA(n_components=6)
pca.fit(cust_prod)
pca_samples = pca.transform(cust_prod)
ps = pd.DataFrame(pca_samples)
print(ps.head(5))

tocluster = pd.DataFrame(ps[[4, 1]])
print(tocluster.shape)
print(tocluster.head())

fig = plt.figure(figsize=(8, 8))
plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='purple', alpha=0.5, label='all data')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()

clusterer = KMeans(n_clusters=4, random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
print(centers)

print(c_preds[0:100])

fig = plt.figure(figsize=(8, 8))
colors = ['gold', 'lightpink', 'mediumpurple', 'powderblue']
colored = [colors[k] for k in c_preds]
print(colored[0:10])
plt.scatter(tocluster[4], tocluster[1],  color=colored)

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()

clust_prod = cust_prod.copy()
clust_prod['cluster'] = c_preds

c0 = clust_prod[clust_prod['cluster'] == 0].drop('cluster', axis=1).mean()
c1 = clust_prod[clust_prod['cluster'] == 1].drop('cluster', axis=1).mean()
c2 = clust_prod[clust_prod['cluster'] == 2].drop('cluster', axis=1).mean()
c3 = clust_prod[clust_prod['cluster'] == 3].drop('cluster', axis=1).mean()

print(c0.sort_values(ascending=False)[0:10])
print(c1.sort_values(ascending=False)[0:10])
print(c2.sort_values(ascending=False)[0:10])
print(c3.sort_values(ascending=False)[0:10])

end_time = datetime.datetime.now()
print("耗时（单位：秒）：")
print((end_time - start_time).seconds)