import AU_EMO_bayes
from AU_EMO_bayes import initGraph
from InferPGM.inference import VariableElimination

from materials.process_priori import cal_interAUPriori

if __name__ == '__main__':
    AU_evidence = { 'AU1':1, 
                    'AU2':0,
                    'AU4':1,
                    'AU5':0,
                    'AU6':0,
                    'AU7':1,
                    'AU9':1,
                    'AU10':0,
                    'AU11':0,
                    'AU12':0,
                    'AU15':0,
                    'AU17':0,
                    'AU20':0,
                    'AU23':0,
                    'AU24':0,
                    'AU25':1,
                    'AU26':1 }
    
    EMO2AU_cpt, prob_AU, EMO_img_num, AU_cpt, EMO, AU = cal_interAUPriori()
    AU_EMO_model = AU_EMO_bayes.AU_EMO_bayesGraph(EMO2AU_cpt, prob_AU, EMO, AU)
    q = AU_EMO_model.infer(AU_evidence)

    # AU_EMO_model = initGraph()
    # model_infer = VariableElimination(AU_EMO_model)
    # q = model_infer.query(variables=['EMO'], evidence=AU_evidence)


    print(q)

    
