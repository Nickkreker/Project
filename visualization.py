import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import cv2
import os


st.title("""Прогноз коффициента полнодревесности и плотного объема""")
path = st.selectbox('Выбор машины', options=['2020_01/09/212845_О376ТН47',
                                             '2020_01/21/224923_М636МВ10',
                                             '2020_01/05/062112_м475рр10',
                                             '2020_01/15/161657_М493ЕЕ10',
                                             '2020_01/06/131352_М136НС10',
                                             '2020_01/30/173809_В712УМ47'])

path = path.replace('2020_01', 'part_1')
df_pack = pd.read_csv('pack.csv')
df_pack = df_pack[df_pack['path'].apply(lambda x: x[:25]) == path]
num_frames = len(os.listdir(f'{path}/FrontJPG/'))


path = path.replace('part_1', 'scan_track_video/2020_01')



@st.cache
def load_nek(path):
    nek = pd.read_csv('nek_streamlit.csv')
    nek = nek[nek['path'] == path]
    ids = nek['pack_id']
    return nek, ids

@st.cache
def load_nek_features(ids):
    nek_features = pd.read_csv('feature_table_nek.csv')
    nek_features = nek_features[nek_features['pack_id'] in ids]
    return nek_features



nek, ids = load_nek(path)
st.write('**Дата:**', nek['datetime'].iloc[0][:10], '--',
         '**Время**', nek['datetime'].iloc[0][11:], '--',
         '**Номер машины**', path[-8:].lower())

# Слайдер по фотографиям машины
path = path.replace('scan_track_video/2020_01', 'part_1')
frames_filter = st.slider('Номер фото', 0, num_frames - 1, 0)
df_pack['sort'] = df_pack['path'].apply(lambda x: x[-10:]).str.extract('(\d+)', expand=False).astype(int)
df_pack.sort_values('sort',inplace=True, ascending=True)

chart = alt.Chart(df_pack[['sort', 'is_pack']], height=50, width=700).mark_line(point=True).encode(
    alt.X('sort', axis=None),
    alt.Y('is_pack', axis=None)
)
vline = alt.Chart(pd.DataFrame({'selected_frame': [frames_filter]}), width=700).mark_rule(color='red').encode(
    alt.X('selected_frame', axis=None)
)
st.altair_chart(alt.layer(chart, vline))


# Отображение выбранного frame
stream = open(f'{path}/FrontJPG/front{frames_filter}.jpg', "rb")
bytes = bytearray(stream.read())
numpyarray = np.asarray(bytes, dtype=np.uint8)
img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape
cv2.line(img, (750, 180), (750, 970), (0, 255, 0), thickness=2)
cv2.line(img, (1400, 180), (1400, 970), (0, 255, 0), thickness=2)
st.image(img, use_column_width=True)

path = path.replace('part_1', 'scan_track_video/2020_01')





st.subheader("""Список пачек и данные НЭК""")
# frames, sort, length, width, height, k, volume
st.dataframe(nek[['frames', 'sort', 'length', 'width', 'height', 'k', 'volume']])

st.header("""Прогнозы моделей""")
st.subheader('Сорт')
model_acc = pd.read_csv('sort_accuracy.csv')
st.dataframe(model_acc.style.highlight_max(axis=0), width=1000)

st.subheader('КПД и плотный объем')
kpd_metrics = pd.read_csv('kpd_metrics.csv')
st.dataframe(kpd_metrics)
