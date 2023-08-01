function [F P R nmi avgent AR ACC] = baseline_acc(pred,truth)
    ACC = Accuracy(pred, truth);
    [A nmi avgent] = compute_nmi(truth,pred);
    if (min(truth)==0)
        [AR,RI,MI,HI]=RandIndex(truth+1,pred);
    else
        [AR,RI,MI,HI]=RandIndex(truth,pred);
    end  
    [F,P,R] = compute_f(truth,pred);
end