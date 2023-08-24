from autogluon.tabular import TabularDataset, TabularPredictor
import datetime
import pandas as pd


def run(
    train_cols = [],
    test_cols = [],
    save_results = True,
):
    train = TabularDataset('data/train.csv')[train_cols]
    test = TabularDataset('data/test.csv')[test_cols]
    
    predictor = TabularPredictor(label='label').fit(train_data=train)
    predictions = predictor.predict(test)
    
    print(predictor.leaderboard(train, silent=True).sort_values(by=['score_val'], axis=0, ascending=False))
    
    if save_results:
        submit = pd.read_csv('data/sample_submission.csv')
        # 예측한 값을 TARGET 컬럼에 할당합니다.
        submit['label'] = predictions
        submit.head()
        # 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
        now = datetime.datetime.now()
        time_now = now.strftime('%Y-%m-%d_%H-%M-%S') 
        submit.to_csv(f'results/autogluon_submission_{str(time_now)}.csv', index = False)
    

if __name__ == '__main__':
    test_cols = [
       # '나이', 
       '키(cm)', 
       '몸무게(kg)', 
       # 'BMI', 
       '시력', 
       # '충치', 
       # '공복 혈당', 
       '혈압',
       # '중성 지방', 
       # '혈청 크레아티닌', 
       '콜레스테롤', 
       '고밀도지단백', 
       '저밀도지단백', 
       '헤모글로빈', 
       '요 단백',
       # '간 효소율'
    ]
    train_cols = test_cols + ['label']
    run(train_cols=train_cols, test_cols=test_cols, save_results=True)