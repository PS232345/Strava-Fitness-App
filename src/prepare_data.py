import pandas as pd, numpy as np
d='data/raw/'; o='data/'
dt=lambda s: pd.to_datetime(s, format='%m/%d/%Y %I:%M:%S %p')
dd=lambda s: pd.to_datetime(s, format='%m/%d/%Y')
log=[]

# ---- daily
da=pd.read_csv(d+'dailyActivity_merged.csv'); da['Date']=dd(da.ActivityDate); da=da.drop(columns='ActivityDate')
da['IsNonWearDay']=((da.TotalSteps==0)&(da.SedentaryMinutes==1440)).astype(int)
sl=pd.read_csv(d+'sleepDay_merged.csv'); n=len(sl); sl=sl.drop_duplicates(); log.append(('sleepDay exact duplicates removed',n-len(sl)))
sl['Date']=dt(sl.SleepDay).dt.normalize(); sl=sl.drop(columns='SleepDay')
w=pd.read_csv(d+'weightLogInfo_merged.csv'); w['Date_full']=dt(w.Date); w['Date']=w.Date_full.dt.normalize()
w=w.sort_values('Date_full').groupby(['Id','Date'],as_index=False).last()
w=w[['Id','Date','WeightKg','WeightPounds','BMI','IsManualReport']]  # Fat dropped: 65/67 missing

# ---- heart rate -> minute
hr=pd.read_csv(d+'heartrate_seconds_merged_1.csv'); hr['Time']=dt(hr.Time)
hr['Minute']=hr.Time.dt.floor('min')
hrm=hr.groupby(['Id','Minute']).Value.mean().round(1).rename('HR_mean').reset_index()
hrh=hr.assign(Hour=hr.Time.dt.floor('h')).groupby(['Id','Hour']).Value.agg(HR_mean='mean',HR_min='min',HR_max='max',HR_readings='count').round(1).reset_index()
hrd=hr.assign(Date=hr.Time.dt.normalize()).groupby(['Id','Date']).Value.agg(HR_mean='mean',HR_min='min',HR_max='max',HR_readings='count').round(1).reset_index()
del hr

# ---- minute narrow
mc=pd.read_csv(d+'minuteCaloriesNarrow_merged_1.csv'); mi=pd.read_csv(d+'minuteIntensitiesNarrow_merged_1.csv')
mm=pd.read_csv(d+'minuteMETsNarrow_merged_1.csv'); ms=pd.read_csv(d+'minuteStepsNarrow_merged_1.csv')
for x in (mc,mi,mm,ms): x['Minute']=dt(x.ActivityMinute); x.drop(columns='ActivityMinute',inplace=True)
mn=mc.merge(mi,on=['Id','Minute']).merge(mm,on=['Id','Minute']).merge(ms,on=['Id','Minute'])
assert len(mn)==len(mc)
mn['METs']=mn.METs/10   # Fitbit stores METs x10
mn['Calories']=mn.Calories.round(4)
sm=pd.read_csv(d+'minuteSleep_merged.csv'); n=len(sm); sm=sm.drop_duplicates(); log.append(('minuteSleep duplicates removed',n-len(sm)))
sm['Minute']=dt(sm.date); sm=sm.rename(columns={'value':'SleepState','logId':'SleepLogId'}).drop(columns='date')
sm=sm.drop_duplicates(['Id','Minute'])
mn=mn.merge(sm,on=['Id','Minute'],how='left').merge(hrm,on=['Id','Minute'],how='left')
mn['SleepState']=mn.SleepState.astype('Int8')  # 1 asleep, 2 restless, 3 awake; blank = not in sleep log
mn=mn.sort_values(['Id','Minute'])
mn.to_csv(o+'fitbit_minute_merged.csv',index=False)

# ---- hourly
hc=pd.read_csv(d+'hourlyCalories_merged.csv'); hi=pd.read_csv(d+'hourlyIntensities_merged.csv'); hs=pd.read_csv(d+'hourlySteps_merged.csv')
hh=hc.merge(hi,on=['Id','ActivityHour']).merge(hs,on=['Id','ActivityHour']); hh['Hour']=dt(hh.ActivityHour); hh=hh.drop(columns='ActivityHour')
mh=mn.assign(Hour=mn.Minute.dt.floor('h')).groupby(['Id','Hour']).agg(METs_mean=('METs','mean'),MinutesAsleep=('SleepState',lambda s:(s==1).sum())).round(3).reset_index()
hh=hh.merge(mh,on=['Id','Hour'],how='left').merge(hrh,on=['Id','Hour'],how='left')
hh=hh[['Id','Hour']+[c for c in hh.columns if c not in('Id','Hour')]].sort_values(['Id','Hour'])
hh.to_csv(o+'fitbit_hourly_merged.csv',index=False)

# ---- daily merge
daily=da.merge(sl,on=['Id','Date'],how='left').merge(w,on=['Id','Date'],how='left').merge(hrd,on=['Id','Date'],how='left')
md=mn.assign(Date=mn.Minute.dt.normalize()).groupby(['Id','Date']).METs.mean().round(3).rename('METs_mean').reset_index()
daily=daily.merge(md,on=['Id','Date'],how='left').sort_values(['Id','Date'])
daily.to_csv(o+'fitbit_daily_merged.csv',index=False)

print(log)
for n_,x in [('daily',daily),('hourly',hh),('minute',mn)]:
    print(n_,x.shape); print(x.isna().sum()[x.isna().sum()>0].to_dict())
print(daily.IsNonWearDay.sum(), daily.Date.min(), daily.Date.max())
