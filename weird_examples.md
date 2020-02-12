# `FillerOrderNo`s for Werid Appointments worth Investigating

```python

```

More than one transition into Examined

`pd.pivot_table(df, index='FillerOrderNo', columns='now_status', values='date', aggfunc='count').sort_values('examined', ascending=False)`

```python
5708150
5760317
```

### Weird Transitions
```python
df[df['NoShow']].groupby(['was_status', 'now_status'])['FillerOrderNo'].count().sort_values()
df[(df['NoShow']) & (df['was_status'] == 'examined')]
```

```python
was_status  now_status
waiting     scheduled      2
registered  scheduled      5
scheduled   scheduled     11
```

went from Waiting -> Scheduled
```python
fon = 5224916
fon = 5751777
```

went from Waiting -> Cancelled
```python
fon = 5653544
```



## Resolved
```python
fon = 5713876  # 4 start-examined changes within 1 minute, fine
```
