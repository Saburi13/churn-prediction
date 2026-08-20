import pandas as pd

def main(path='data/clean_telco.csv'):
    df = pd.read_csv(path)
    insights = []

    # 1. Target distribution
    # normalize churn to 0/1 numeric if needed
    if df['Churn'].dtype == object:
        # assume 'Yes'/'No'
        df['Churn_bool'] = df['Churn'].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    else:
        df['Churn_bool'] = df['Churn'].astype(int)
    counts = df['Churn_bool'].value_counts()
    pct_yes = counts.get(1, 0) / counts.sum() * 100
    insights.append(f"Churn rate: {pct_yes:.2f}% (Yes: {int(counts.get(1,0))}, No: {int(counts.get(0,0))})")

    # 2. TotalCharges dtype / issues
    tc_dtype = df['TotalCharges'].dtype
    na_tc = df['TotalCharges'].isna().sum()
    insights.append(f"TotalCharges dtype is {tc_dtype}; missing after cleaning: {na_tc}")

    # 3. Tenure distribution and churn by short tenure
    ten_median = df['tenure'].median()
    short = df[df['tenure']<=6]
    short_churn_rate = (short['Churn_bool']==1).mean()*100
    insights.append(f"Median tenure: {ten_median}. Customers with tenure<=6 months churn at {short_churn_rate:.2f}%")

    # 4. MonthlyCharges distribution
    mc_median = df['MonthlyCharges'].median()
    high_mc = df[df['MonthlyCharges']>=mc_median]
    high_mc_churn = (high_mc['Churn_bool']==1).mean()*100
    insights.append(f"Median MonthlyCharges: {mc_median:.2f}. Customers above median churn at {high_mc_churn:.2f}%")

    # 5. Contract type churn rates
    contract = df.groupby('Contract')['Churn_bool'].value_counts(normalize=True).unstack().fillna(0)
    contract_insights = []
    for c in contract.index:
        churn_pct = contract.loc[c].get('Yes', 0)*100
        contract_insights.append(f"{c}: {churn_pct:.2f}% churn")
    insights.append("Contract churn rates: " + "; ".join(contract_insights))

    for i,ins in enumerate(insights,1):
        print(f"{i}. {ins}")

if __name__ == '__main__':
    main()
