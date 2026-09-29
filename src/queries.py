"""Named SQL analyses (SQLite dialect). key -> (title, question, sql)."""
NONWEAR = "IsNonWearDay = 0"
QUERIES = {
"completeness": ("Data completeness", "How much of each signal do we actually have?", """
SELECT 'Steps/activity (worn days)' AS metric, COUNT(DISTINCT Id) AS users, ROUND(100.0*SUM(IsNonWearDay=0)/COUNT(*),1) AS pct_user_days FROM daily
UNION ALL SELECT 'Sleep', COUNT(DISTINCT CASE WHEN TotalMinutesAsleep IS NOT NULL THEN Id END), ROUND(100.0*SUM(TotalMinutesAsleep IS NOT NULL)/COUNT(*),1) FROM daily
UNION ALL SELECT 'Heart rate', COUNT(DISTINCT CASE WHEN HR_mean IS NOT NULL THEN Id END), ROUND(100.0*SUM(HR_mean IS NOT NULL)/COUNT(*),1) FROM daily
UNION ALL SELECT 'Weight/BMI', COUNT(DISTINCT CASE WHEN WeightKg IS NOT NULL THEN Id END), ROUND(100.0*SUM(WeightKg IS NOT NULL)/COUNT(*),1) FROM daily"""),

"segments": ("User segments by activity level", "Who are the sedentary vs active users, and how do they sleep?", """
WITH u AS (SELECT Id, AVG(TotalSteps) s, AVG(Calories) c, AVG(TotalMinutesAsleep) sl, AVG(VeryActiveMinutes) va
           FROM daily WHERE IsNonWearDay=0 GROUP BY Id)
SELECT CASE WHEN s<5000 THEN '1 Sedentary (<5k)' WHEN s<7500 THEN '2 Low active (5-7.5k)'
            WHEN s<10000 THEN '3 Somewhat active (7.5-10k)' ELSE '4 Active (10k+)' END AS segment,
       COUNT(*) AS users, ROUND(AVG(s)) AS avg_steps, ROUND(AVG(va),1) AS avg_very_active_min,
       ROUND(AVG(c)) AS avg_calories, ROUND(AVG(sl)) AS avg_sleep_min
FROM u GROUP BY segment ORDER BY segment"""),

"weekday": ("Weekday pattern", "Which days are people most active and best rested?", f"""
SELECT strftime('%w',Date) AS dow,
       CASE strftime('%w',Date) WHEN '0' THEN 'Sun' WHEN '1' THEN 'Mon' WHEN '2' THEN 'Tue' WHEN '3' THEN 'Wed'
            WHEN '4' THEN 'Thu' WHEN '5' THEN 'Fri' ELSE 'Sat' END AS weekday,
       COUNT(*) AS user_days, ROUND(AVG(TotalSteps)) AS avg_steps,
       ROUND(AVG(VeryActiveMinutes),1) AS avg_very_active_min, ROUND(AVG(TotalMinutesAsleep)) AS avg_sleep_min
FROM daily WHERE {NONWEAR} GROUP BY dow ORDER BY dow"""),

"sleep_vs_steps": ("Sleep by step band", "Do more active days go with longer or better sleep?", f"""
SELECT CASE WHEN TotalSteps<5000 THEN 'a) <5k steps' WHEN TotalSteps<10000 THEN 'b) 5-10k steps' ELSE 'c) 10k+ steps' END AS steps_band,
       COUNT(*) AS nights, ROUND(AVG(TotalMinutesAsleep)) AS avg_sleep_min,
       ROUND(AVG(100.0*TotalMinutesAsleep/TotalTimeInBed),1) AS avg_efficiency_pct
FROM daily WHERE {NONWEAR} AND TotalMinutesAsleep IS NOT NULL GROUP BY steps_band ORDER BY steps_band"""),

"intensity_hr": ("Intensity vs heart rate", "Do high-intensity days show different heart-rate and sleep (recovery proxies)?", f"""
SELECT CASE WHEN VeryActiveMinutes>=30 THEN 'c) High (30+ very-active min)' WHEN VeryActiveMinutes>=10 THEN 'b) Moderate (10-29)' ELSE 'a) Low (<10)' END AS intensity_tier,
       COUNT(*) AS user_days, ROUND(AVG(HR_min),1) AS avg_min_hr, ROUND(AVG(HR_mean),1) AS avg_mean_hr,
       ROUND(AVG(TotalMinutesAsleep)) AS avg_sleep_min
FROM daily WHERE {NONWEAR} AND HR_min IS NOT NULL GROUP BY intensity_tier ORDER BY intensity_tier"""),

"hourly": ("24-hour rhythm", "When during the day do people move, burn calories and peak in heart rate?", """
SELECT CAST(strftime('%H',Hour) AS INTEGER) AS hour_of_day, ROUND(AVG(StepTotal),1) AS avg_steps,
       ROUND(AVG(Calories),1) AS avg_calories, ROUND(AVG(HR_mean),1) AS avg_hr
FROM hourly GROUP BY hour_of_day ORDER BY hour_of_day"""),

"weekly": ("Weekly trend (window function)", "Is activity trending up or down week over week?", f"""
WITH w AS (SELECT strftime('%Y-W%W',Date) AS week, ROUND(AVG(TotalSteps)) AS avg_steps, COUNT(*) AS user_days
           FROM daily WHERE {NONWEAR} GROUP BY week)
SELECT week, user_days, avg_steps, avg_steps - LAG(avg_steps) OVER (ORDER BY week) AS change_vs_prev_week FROM w"""),

"compliance": ("Wear compliance ranking", "Which users wore the tracker consistently? (RANK window function)", """
SELECT Id, COUNT(*) AS days_logged, SUM(IsNonWearDay) AS nonwear_days,
       ROUND(100.0*SUM(IsNonWearDay)/COUNT(*),1) AS nonwear_pct,
       RANK() OVER (ORDER BY SUM(IsNonWearDay), COUNT(*) DESC) AS compliance_rank
FROM daily GROUP BY Id ORDER BY compliance_rank LIMIT 15"""),

"sleep_debt": ("Sleep-debt watchlist", "Which users sleep under 7 hours most nights?", """
SELECT Id, COUNT(*) AS nights, ROUND(AVG(TotalMinutesAsleep)) AS avg_sleep_min,
       ROUND(100.0*SUM(TotalMinutesAsleep<420)/COUNT(*)) AS pct_nights_under_7h
FROM daily WHERE TotalMinutesAsleep IS NOT NULL GROUP BY Id HAVING nights>=10 ORDER BY pct_nights_under_7h DESC"""),
}
