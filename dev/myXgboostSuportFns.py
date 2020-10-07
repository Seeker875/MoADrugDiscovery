
### Author Taran
### Xgboost support modelling functions
import pandas as pd
import numpy as np


def GridParamTune(dtrain,gridParams,EvalMetric,params,Drop=1,verbose=1):
    '''
    Xgboost Param tuning
    Author: Taran
    
    Given a params dictionary this function implements Grid and random search
    The cv object can be replaced for any other model
    
    Args - dTrain matrix,
            gridParams A parameters dictinary with candidate search space
            Drop Rate - for  Random search [0,1]
            verbose 0 0r 1 
            params  ---not to be tuned parameters
            EvalMetric  -- takes 1 eval metric
    Output
    Returns a df with results for each params
    Additional dependency itertools
    
    '''
    import itertools
    #paramers passed
    paramNames = list(gridParams.keys())
    
    results = []
    ### iterate over all combinations
    for row in itertools.product(*gridParams.values()):
        ### random search threshold
        if np.random.random(1)[0] > Drop:
            continue 
        
        # insert values into param dict
        for i in range(len(row)):
            params[paramNames[i]] = row[i]
            
        # train model for given params    
        cvN = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    seed=42,
                    nfold=5,
                    metrics={EvalMetric},
                    early_stopping_rounds=25
                )
        #get results 
        bestRound= cvN[f'test-{EvalMetric}-mean'].argmin()
        trainMetric = cvN[f'train-{EvalMetric}-mean'][bestRound]
        testMetric =cvN[f'test-{EvalMetric}-mean'][bestRound]
        overfit = ((cvN[f'test-{EvalMetric}-mean'][bestRound]/cvN[f'train-{EvalMetric}-mean'][bestRound]) -1)*100
        
        #unlist
        tempResults=[list(params.values())[1:],bestRound,trainMetric,testMetric,overfit]      

        results.append(list(itertools.chain.from_iterable(i if isinstance(i, list) else [i] for i in tempResults)))
         
    colNames=[paramNames,'bestRound',f'train-{EvalMetric}',f'test-{EvalMetric}','overfit']              
    df = pd.DataFrame(results,columns=list(itertools.chain.from_iterable(i if isinstance(i, list) else [i] for i in colNames)))
                  
    return df

def overFitRelations(paramName,paramResults):
    '''
    Arg - paramName
    returns a plot and a df showing over fit relation
    '''
    df = paramResults.groupby([paramName])['overfit'].mean()
    print(df.plot.bar(title=f'{paramName} vs overfitting'))
    return df

def getLogLoss(y,yhat):
    '''
    logloss
    '''
    assert len(y)==len(yhat)
    eps= 1e-12
    
    yhat= np.clip(yhat,eps,1-eps)
    return -1*np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))


def PreProcessX(df):
    '''
    Preprocessing for independent  vars
    encode categoricals
    
    returns processed df,
    '''
    df['cp_dose'] = (df['cp_dose'] == 'D1').astype(int)
    df['cp_type'] = (df['cp_type'] == 'trt_cp').astype(int)
    
    return df

def flattenList(myList):
    '''
    Flattens a list and returns it
    '''
    import itertools
    return list(itertools.chain.from_iterable(i if isinstance(i, list) else [i] for i in myList))

def dropCorrelatedVars(corrM,cutoff):
    '''
    drop correlated vars having cor > cutoff, of the 2 variables with high corr
    will drop the one with the lower importance
    corrM corrlation matrix in order of least important to most imp variables
    cutoff Threshold
    '''
    
    dropVars=[]
    for i in range(corrM.shape[0]):
        for j in range(i+1, corrM.shape[0]):

            if corrM.iloc[i,j] >cutoff:
                #print(f' corr of {corrM.columns[i],corrM.columns[j]} is {corrM.iloc[i,j]}')
                dropVars.append(corrM.columns[j])
                
    return list(set(dropVars))

# dropping least important variables 
def dropVars(sVars,end,st=1,step=1):
    '''
    sVars variable list starting from least important variables
    
    drop variables upto end increasingly from st to end
    for large number of variables step size should be increased, 
    dropping step number of variables at a time
    
    returns result dropping variables
    '''
    results ={}
    for i in range(st,end,step):
        
        remainingVars = sVars[i:] # drop variable
        dtrain = xgb.DMatrix(xTrainVars[remainingVars], label=yTrainVars)
        mod = xgb.train(params1,dtrain, num_boost_round=978) ## params1 are same selected variables
        rLoss = getLogLoss(yTestVars,mod.predict(xgb.DMatrix(xTestVars[remainingVars]))[:,1]) 
        results[f'dropped_{i}'] = (rLoss,(testLoss-rLoss))
    
    return results

def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    '''
    Target encoding for test data

    '''
    # Calculate global mean on the train data
    global_mean = train[target].mean()

    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()

    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values


def train_mean_target_encoding(train, target, categorical, alpha=5):
    '''
    5 fold for target encoding
    encode values for a fold from other 4

    '''
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature       
    return train_feature.values


def mean_target_encoding(train, test, target, categorical, alpha=5):
    '''
    Final fn for encoding Target mean
    '''
  
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
  
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    
    # Return new features to add to the model
    return train_feature, test_feature

