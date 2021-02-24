import collections
'''
def _evaluate_acc_f1(self, data_loader):
    n_correct, n_total = 0, 0
    t_targets_all, t_outputs_all = None, None
    # switch model to evaluation mode
    self.model.eval()
    with torch.no_grad():
        #t_batch:40,t_sample_batched:データをまとめたやつ
        for t_batch, t_sample_batched in enumerate(data_loader):
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            t_targets = t_sample_batched['polarity'].to(self.opt.device)
            t_outputs = self.model(t_inputs)
            n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
            n_total += len(t_outputs)
            for i,a_list in enumerate(t_sample_batched['aspect_in_text'].to(self.opt.device)):
                if len(a_list)>=3:
                    print(i,':',a_list)
            if t_targets_all is None:
                t_targets_all = t_targets
                t_outputs_all = t_outputs
            else:
                t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        
    answer_list = t_targets_all.detach().clone().numpy().tolist()
    result_list = torch.argmax(t_outputs_all,-1).detach().clone().numpy().tolist()
    with open('answer.txt', mode = 'w') as f:
        for i in range(0,len(answer_list)):
            f.write(str(answer_list[i]) + '\n')
    with open('result.txt',mode = 'w') as f:
        for i in range(0,len(result_list)):   
            f.write(str(result_list[i])+'\n')

    acc = n_correct / n_total
    f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
    return acc, f1
'''

if __name__ == '__main__':
    fname = 'restaurant'
    if fname == 'restaurant':
        file_name = './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    elif fname == 'laptop':
        file_name = './datasets/semeval14/Laptops_Test_Gold.xml.seg'
    else:
        file_name = './datasets/acl-14-short-data/test.raw'
    '''
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }    
    '''
    '''
    text_count = []
    with open(fname+'.txt') as f:
        raw_text_list = [s.strip() for s in f.readlines()]
        c = collections.Counter(raw_text_list)
        raw_text_list = sorted(set(raw_text_list),key = raw_text_list.index)
    with open(fname+'1.txt',mode = 'w') as f:
        for i in range(0,len(raw_text_list)):
            f.write(str(c[raw_text_list[i]]) +' '+ raw_text_list[i]+'\n')
            text_count.append(c[raw_text_list[i]])
        c_count = collections.Counter(text_count)
        print(c_count)
    '''
    answer = []
    with open('result_'+fname+'_test.txt') as f:
        result = [s.strip() for s in f.readlines()]
    with open(file_name) as f:
        answer_list = [s.strip() for s in f.readlines()]
    for i in range(0,len(answer_list)):
        if i % 3 == 2:
            answer.append(int(answer_list[i])+1)
    with open(fname+'_test.txt') as f:
        text = [s.strip() for s in f.readlines()]
    with open(fname+'_test_aspect.txt') as f:
        aspect = [s.strip() for s in f.readlines()]
    aspect_num_text,aspect_num = zip(*collections.Counter(text).most_common())

    '''
    #アスペクトの出現頻度調査
    aspect_freq_dic = collections.Counter(aspect).most_common()
    aspect_dic,aspect_freq = zip(*aspect_freq_dic)
    aspect_pfreq,aspect_nfreq,aspect_efreq = [],[],[]
    aspect_non_target = []
    total_aspect_index = []
    aspect_text,non_aspect_text = [],[]
    for index in range(0,len(aspect_dic)):
        aspect_pfreq_num,aspect_nfreq_num,aspect_efreq_num = 0,0,0
        for j in range(0,len(aspect)):
            if aspect_dic[index] == aspect[j]:
                if answer[j] == 2:
                    aspect_pfreq_num += 1
                if answer[j] == 1:
                    aspect_efreq_num += 1
                if answer[j] == 0:
                    aspect_nfreq_num += 1
        aspect_pfreq.append(aspect_pfreq_num)
        aspect_nfreq.append(aspect_nfreq_num)
        aspect_efreq.append(aspect_efreq_num)
    for index in range(0,len(aspect_dic)):
        part_aspect_index = []
        aspect_text_part,non_aspect_text_part = [],[]
        for j,content in enumerate(text):
            if len(aspect_dic[index].split()) >= 2:
                if aspect_dic[index] in content:
                    part_aspect_index.append(j)
                    if aspect[j] == aspect_dic[index]:
                        aspect_text_part.append(content)
                    else:
                        non_aspect_text_part.append(content)
            else:
                for k,k_content in enumerate(content.split()):
                    if aspect_dic[index] == k_content:
                        part_aspect_index.append(j)
                        if aspect[j] == aspect_dic[index]:
                            aspect_text_part.append(content)
                        else:
                            non_aspect_text_part.append(content)
        total_aspect_index.append(part_aspect_index)
        aspect_text_part = list(set(aspect_text_part))
        non_aspect_text_part = list(set(non_aspect_text_part))
        for i,content in enumerate(aspect_text_part):
            if  content in non_aspect_text_part:
                non_aspect_text_part.remove(aspect_text_part[i])
        aspect_text.append(aspect_text_part)
        non_aspect_text.append(non_aspect_text_part)
        aspect_non_target.append(len(non_aspect_text_part)) 
    '''
    '''
    with open(fname+'_aspect_emotion_text.txt',mode = 'w')as f:
        for i in range(0,len(aspect_dic)):
            f.write('アスペクト：'+aspect_dic[i]+'\n')
            f.write('アスペクトとなる文：\n')
            for j in range(0,len(aspect_text[i])):
                for k in range(0,len(text)):
                    if aspect[k] == aspect_dic[i] and text[k] == aspect_text[i][j]:
                        f.write('('+str(answer[k]) +') '+aspect_text[i][j]+'\n')    
    with open(fname+'_neutral_vs_nontarget.txt',mode = 'w')as f:
        for i in range(0,len(aspect_dic)):
            f.write('アスペクト：'+aspect_dic[i]+'\n')
            f.write('アスペクトとなる文：\n')
            for j in range(0,len(aspect_text[i])):
                f.write(aspect_text[i][j]+'\n')
            f.write('アスペクトとならない文：\n')
            for j in range(0,len(non_aspect_text[i])):
                f.write(non_aspect_text[i][j]+'\n')
            f.write('---------------------------------------------\n')    
    with open(fname +'_aspect_frequency_list.tsv',mode = 'w') as f:
        for i in range(0,len(aspect_dic)):
            f.write(aspect_dic[i]+'\t'+str(aspect_freq[i])+'\t'+str(aspect_pfreq[i])+'\t'+str(aspect_nfreq[i])+'\t'+str(aspect_efreq[i])+'\n')
    ''' 
    
    '''
    #twitterのfact・opinion調査
    with open(fname+'_positive_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            for j in range(0,len(text)):
                if int(aspect_num[i]) == 1 and text[j] == aspect_num_text[i]:
                    if answer[j] == 2 :
                        f.write('['+aspect[j]+']'+'\n'+text[j]+'\n') 
    with open(fname+'_negative_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            for j in range(0,len(text)):
                if int(aspect_num[i]) == 1 and text[j] == aspect_num_text[i]:
                    if answer[j] == 0:
                        f.write('['+aspect[j]+']'+'\n'+text[j]+'\n') 
    with open(fname + '_positive_neutral_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            for j in range(0,len(text)):
                if int(aspect_num[i]) == 1 and text[j] == aspect_num_text[i]:
                    if answer[j] == 2 and result[j] == '1':
                        f.write('['+aspect[j]+']'+'\n'+text[j]+'\n')                     
    with open(fname + '_negative_neutral_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            for j in range(0,len(text)):
                if int(aspect_num[i]) == 1 and text[j] == aspect_num_text[i]:
                    if answer[j] == 0 and result[j] == '1':
                        f.write('['+aspect[j]+']'+'\n'+text[j]+'\n')
    '''

    '''
    #neutral　調査
    with open(fname + '_neutral_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            for j in range(0,len(text)):
                if int(aspect_num[i]) == 1 and text[j] == aspect_num_text[i]:
                    if answer[j] == 1:
                        f.write('['+aspect[j]+']'+'\n'+text[j]+'\n')
    with open(fname + '_neutral_positive_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            if int(aspect_num[i]) == 2:
                need_list1 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
                if len(need_list1) == 2:
                    if answer[need_list1[0]] == 1 and answer[need_list1[1]] == 2:
                        f.write('['+aspect[need_list1[0]]+':'+str(answer[need_list1[0]])+','+aspect[need_list1[1]]+':'+str(answer[need_list1[1]])+']'+'\n')
                        f.write(text[need_list1[0]]+'\n')
                    if answer[need_list1[0]] == 2 and answer[need_list1[1]] == 1:
                        f.write('['+aspect[need_list1[0]]+':'+str(answer[need_list1[0]])+','+aspect[need_list1[1]]+':'+str(answer[need_list1[1]])+']'+'\n')
                        f.write(text[need_list1[0]]+'\n')                      
    with open(fname + '_neutral_negative_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            if int(aspect_num[i]) == 2:
                need_list2 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
                if len(need_list2) == 2:
                    if answer[need_list2[0]] == 1 and answer[need_list2[1]] == 0:
                        f.write('['+aspect[need_list2[0]]+':'+str(answer[need_list2[0]])+','+aspect[need_list2[1]]+':'+str(answer[need_list2[1]])+']'+'\n')
                        f.write(text[need_list2[0]]+'\n')
                    if answer[need_list2[0]] == 0 and answer[need_list2[1]] == 1:
                        f.write('['+aspect[need_list2[0]]+':'+str(answer[need_list2[0]])+','+aspect[need_list2[1]]+':'+str(answer[need_list2[1]])+']'+'\n')
                        f.write(text[need_list2[0]]+'\n')
    with open(fname+'_multi_neutral.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            if int(aspect_num[i]) >= 3:
                need_list3 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
                judge = False
                for j in range(0,len(need_list3)):
                    if answer[need_list3[j]] == 1:
                        judge = True
                if judge:
                    for j in range(0,len(need_list3)):
                        f.write('['+aspect[need_list3[j]]+':'+str(answer[need_list3[j]])+'],')
                    f.write('\n'+text[need_list3[0]]+'\n')
    with open(fname + '_double_neutral_text.txt',mode = 'w') as f:
        for i in range(0,len(aspect_num)):
            if int(aspect_num[i]) == 2:
                need_list4 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
                if len(need_list4) == 2:
                    if answer[need_list4[0]] == 1 and answer[need_list4[1]] == 1:
                        f.write('['+aspect[need_list4[0]]+':'+str(answer[need_list4[0]])+','+aspect[need_list4[1]]+':'+str(answer[need_list4[1]])+']'+'\n')
                        f.write(text[need_list4[0]]+'\n')
    '''
    '''
    #意見or事実　別実験結果
    correct_count1,correct_count2,correct_count3,correct_count4 = 0,0,0,0
    #print('-------------------------------------------------------------------')    
    for i in range(0,len(aspect_num)):
        for j in range(0,len(text)):
            if int(aspect_num[i]) == 1 and text[j] == aspect_num_text[i]:
                if answer[j] == 1 and result[j] == '1':
                    correct_count1 += 1
                    #print('('+aspect[j]+') '+text[j])
    print('-------------------------------------------------------------------')    
    for i in range(0,len(aspect_num)):
        if int(aspect_num[i]) == 2:
            need_list1 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            if len(need_list1) == 2:
                if answer[need_list1[0]] == 1 and answer[need_list1[1]] == 2 and result[need_list1[0]] == '1' :
                    correct_count2 += 1
                    print('('+aspect[need_list1[0]]+','+aspect[need_list1[1]]+') '+text[need_list1[0]])
                if answer[need_list1[0]] == 2 and answer[need_list1[1]] == 1 and result[need_list1[1]] == '1':
                    correct_count2 += 1
                    print('('+aspect[need_list1[0]]+','+aspect[need_list1[1]]+') '+text[need_list1[0]])
    print('k-------------------------------------------------------------------')
    for i in range(0,len(aspect_num)):
        if int(aspect_num[i]) == 2:
            need_list2 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            if len(need_list2) == 2:
                if answer[need_list2[0]] == 1 and answer[need_list2[1]] == 0 and result[need_list2[0]] == '1':
                    correct_count3 += 1
                    #print('('+aspect[need_list2[0]]+','+aspect[need_list2[1]]+') '+text[need_list2[0]])
                if answer[need_list2[0]] == 0 and answer[need_list2[1]] == 1 and result[need_list2[1]] == '1':
                    correct_count3 += 1
                    #print('('+aspect[need_list2[0]]+','+aspect[need_list2[1]]+') '+text[need_list2[0]])
    #print('-------------------------------------------------------------------')
    correct_count5 = 0
    for i in range(0,len(aspect_num)):
        if int(aspect_num[i]) == 2:
            need_list1 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            print_judge1 = False
            if len(need_list1) == 2 and answer[need_list1[0]] == 1 and answer[need_list1[1]] == 1:
                if answer[need_list1[0]] == 1 and result[need_list1[0]] == '1':
                    correct_count5 += 1
                    print('アスペクト：'+aspect[need_list1[0]])
                    print_judge1 = True
                if answer[need_list1[1]] == 1 and result[need_list1[1]] == '1':
                    correct_count5 += 1
                    print('アスペクト：'+aspect[need_list1[1]])
                    print_judge1 = True
            if print_judge1:
                print('テキスト：'+text[need_list1[0]])
                print('-------------------------------------------------------------------')   
    #print('-------------------------------------------------------------------')
    for i in range(0,len(aspect_num)):
        if int(aspect_num[i]) >= 3:
            need_list3 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            judge = False
            for j in range(0,len(need_list3)):
                if answer[need_list3[j]] == 1:
                    judge = True
            if judge:
                print_judge = 0
                for k in range(0,len(need_list3)):
                    if answer[need_list3[k]] == 1 and result[need_list3[k]] == '1':
                        correct_count4 += 1
                        print_judge += 1
                        print('アスペクト：'+aspect[need_list3[k]])
                if print_judge != 0:
                    something = 0
                    print('テキスト：'+text[need_list3[0]])                   
                    print('-------------------------------------------------------------------')
    print('中立とアノテートされたアスペクトに関する調査結果')
    print('アスペクト1個:'+str(correct_count1))
    print('(正、中立)ペア:'+str(correct_count2))
    print('(負、中立)ペア:'+str(correct_count3))
    print('(中立、中立)ペア:'+str(correct_count5))    
    print('アスペクト3個以上:'+str(correct_count4))
    '''
    '''
    #all中立の場合と極性が混ざっている場合の調査(focus_countはall中立)
    only_neutral_total_count,only_neutral_answer_count = 0,0
    total_focus_count,answer_focus_count,total_unfocus_count,answer_unfocus_count = 0,0,0,0
    sent_focus_count,sent_unfocus_count = 0,0
    for i in range(0,len(aspect_num)):
        if aspect_num[i] == 1:
            for index ,content in enumerate(text):
                 if aspect_num_text[i]== content and aspect[index]:
                    as_index = index 
            #全体数カウント
            if answer[as_index] == 1:
                only_neutral_total_count +=1
                #正解数カウント
                if result[as_index] == '1':
                    only_neutral_answer_count += 1
                else:
                    print('['+aspect[as_index]+']/ '+text[as_index])
        if aspect_num[i] > 1:
            total_judge_value = 0
            need_list = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            for j in range(0,len(need_list)):
                if answer[need_list[j]] == 1:
                    total_judge_value += 1
            #2つ以上全て中立なアスペクトを持つ文について
            if total_judge_value == len(need_list):
                #全体数カウント                        
                total_focus_count += total_judge_value
                sent_focus_count += 1
                #正解数カウント
                for k in range(0,len(need_list)):
                    if answer[need_list[k]] == 1 and result[need_list[k]] == '1':
                        answer_focus_count += 1
                    if answer[need_list[k]] == 1 and result[need_list[k]] != '1':
                        print('['+aspect[need_list[k]]+']/ '+text[need_list[k]])
            #極性と中立の両方を含む文について 
            if total_judge_value > 0 and total_judge_value != len(need_list):
                #全体数カウント                     
                total_unfocus_count += total_judge_value
                sent_unfocus_count += 1
                #正解数カウント
                for k in range(0,len(need_list)):
                    if answer[need_list[k]] == 1 and result[need_list[k]] == '1':
                        answer_unfocus_count += 1
                    if answer[need_list[k]] == 1 and result[need_list[k]] != '1':
                        print('['+aspect[need_list[k]]+']/ '+text[need_list[k]])
    print(fname+'に関する実験結果')
    print('中立アスペクトが1つの場合：\n総数　'+str(only_neutral_total_count)+',　正解数　'+str(only_neutral_answer_count))
    print('アスペクトが複数ある場合：')
    print('全て中立なアスペクトの総数　'+str(total_focus_count)+', 正解数　'+str(answer_focus_count)+', 文章数 ' + str(sent_focus_count))
    print('極性混じりアスペクトの総数　'+str(total_unfocus_count)+', 正解数　'+str(answer_unfocus_count)+', 文章数 ' + str(sent_unfocus_count))
    '''
    '''
    #CS seminar用　neutral失敗例探し
    for i in range(0,len(aspect_num)):
        if aspect_num[i] > 1:
            total_judge_value = 0
            need_list = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            for j in range(0,len(need_list)):
                if answer[need_list[j]] == 1:
                    total_judge_value += 1
            #極性と中立の両方を含む文について 
            if total_judge_value > 0 and total_judge_value != len(need_list):
                print_judge = False
                for k in range(0,len(need_list)):
                    if answer[need_list[k]] != int(result[need_list[k]]):
                        print_judge = True
                if print_judge:
                    for k in range(0,len(need_list)):
                        print('['+aspect[need_list[k]]+':('+str(answer[need_list[k]])+','+result[need_list[k]]+')] ')   
                    print(text[need_list[0]])
                    print('-------------------------------------------------------------------') 
    '''
    #CSセミナー用成功データ探し
    for i in range(0,len(aspect_num)):
        if int(aspect_num[i]) == 2:
            need_list1 = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            if len(need_list1) == 2 and answer[need_list1[0]] == 2 and answer[need_list1[1]] == 1:
                if result[need_list1[0]] == '2' and result[need_list1[1]] == '1':
                    print('組み合わせ：正中')
                    print('アスペクト：'+aspect[need_list1[0]]+','+aspect[need_list1[1]])
                    print('テキスト：'+text[need_list1[0]])
                    print('-----------------------------------------')
            if len(need_list1) == 2 and answer[need_list1[0]] == 1 and answer[need_list1[1]] == 2:
                if result[need_list1[0]] == '1' and result[need_list1[1]] == '2':
                    print('組み合わせ：中正')
                    print('アスペクト：'+aspect[need_list1[0]]+','+aspect[need_list1[1]])
                    print('テキスト：'+text[need_list1[0]])
                    print('-----------------------------------------')
            if len(need_list1) == 2 and answer[need_list1[0]] == 0 and answer[need_list1[1]] == 1:
                if result[need_list1[0]] == '0' and result[need_list1[1]] == '1':
                    print('組み合わせ：負中')
                    print('アスペクト：'+aspect[need_list1[0]]+','+aspect[need_list1[1]])
                    print('テキスト：'+text[need_list1[0]])
                    print('-----------------------------------------')
            if len(need_list1) == 2 and answer[need_list1[0]] == 1 and answer[need_list1[1]] == 0:
                if result[need_list1[0]] == '1' and result[need_list1[1]] == '0':
                    print('組み合わせ：中負')
                    print('アスペクト：'+aspect[need_list1[0]]+','+aspect[need_list1[1]])
                    print('テキスト：'+text[need_list1[0]])
                    print('-----------------------------------------')
            if len(need_list1) == 2 and answer[need_list1[0]] == 2 and answer[need_list1[1]] == 0:
                if result[need_list1[0]] == '2' and result[need_list1[1]] == '0':
                    print('組み合わせ：正負')
                    print('アスペクト：'+aspect[need_list1[0]]+','+aspect[need_list1[1]])
                    print('テキスト：'+text[need_list1[0]])
                    print('-----------------------------------------')            
            if len(need_list1) == 2 and answer[need_list1[0]] == 0 and answer[need_list1[1]] == 2:
                if result[need_list1[0]] == '0' and result[need_list1[1]] == '2':
                    print('組み合わせ：負正')
                    print('アスペクト：'+aspect[need_list1[0]]+','+aspect[need_list1[1]])
                    print('テキスト：'+text[need_list1[0]])
                    print('-----------------------------------------')

    '''
    #正解内容
    with open(fname+'_pos_pos.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 2 and result[i] == '2' :
                f.write(aspect[i]+'\n'+text[i]+'\n')
    with open(fname+'_neu_neu.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 1 and result[i] == '1' :
                f.write(aspect[i]+'\n'+text[i]+'\n')
    with open(fname+'_neg_neg.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 0 and result[i] == '0' :
                f.write(aspect[i]+'\n'+text[i]+'\n')
    #間違い内容
    with open(fname+'_pos_neg.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 2 and result[i] == '0' :
                f.write(aspect[i]+'\n'+text[i]+'\n')
    with open(fname+'_pos_neu.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 2 and result[i] == '1' :
                f.write(aspect[i]+'\n'+text[i]+'\n')    
    with open(fname+'_neg_pos.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 0 and result[i] == '2':
                f.write(aspect[i]+'\n'+text[i]+'\n')
    with open(fname+'_neg_neu.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 0 and result[i] == '1' :
                f.write(aspect[i]+'\n'+text[i]+'\n')
    with open(fname+'_neu_pos.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 1 and result[i] == '2' :
                f.write(aspect[i]+'\n'+text[i]+'\n')    
    with open(fname+'_neu_neg.txt',mode='w') as f:
        for i in range(0,len(aspect)):
            if answer[i] == 1 and result[i] == '0':
                f.write(aspect[i]+'\n'+text[i]+'\n')
    '''
    '''
    a0,a1,a2 = 0,0,0
    a0_good,a1_good,a2_good = 0,0,0
    for i  in range(0,len(aspect_num)):
        if int(aspect_num[i]) == 1:
            for index,content in enumerate(text):
                if aspect_num_text[i] == content:
                    if answer[index] == 2:
                        a2+=1
                    if answer[index] == 0:
                        a0+=1
                    if answer[index] == 1:
                        a1+=1
                    if int(result[index]) == answer[index] == 2:
                        a2_good+=1
                    if int(result[index]) == answer[index] == 1:
                        a1_good+=1
                    if int(result[index]) == answer[index] == 0:
                        a0_good+=1
    print(fname+'の分析結果(3通り)：')
    print('正:',a2,',正解数：',a2_good)
    print('負:',a0,',正解数：',a0_good)
    print('中:',a1,',正解数：',a1_good)
    a0_1,a0_2,a1_0,a1_2,a2_0,a2_1,a0_0,a1_1,a2_2 = 0,0,0,0,0,0,0,0,0
    ac0_1,ac0_2,ac1_0,ac1_2,ac2_0,ac2_1,ac0_0,ac1_1,ac2_2 = 0,0,0,0,0,0,0,0,0
    for i in range(0,len(aspect_num)):
        if int(aspect_num[i]) == 2:
            need_list = [index for index,content in enumerate(text) if aspect_num_text[i]== content]
            if len(need_list) == 2:
                if answer[need_list[0]] == 2 and answer[need_list[1]] == 2:
                    a2_2+= 1 
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac2_2 += 1
                if answer[need_list[0]] == 2 and answer[need_list[1]] == 1:
                    a2_1+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac2_1 += 1
                if answer[need_list[0]] == 2 and answer[need_list[1]] == 0:
                    a2_0+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac2_0 += 1
                if answer[need_list[0]] == 1 and answer[need_list[1]] == 2:
                    a1_2+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac1_2 += 1
                if answer[need_list[0]] == 1 and answer[need_list[1]] == 1:
                    a1_1+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac1_1 += 1
                if answer[need_list[0]] == 1 and answer[need_list[1]] == 0:
                    a1_0+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac1_0 += 1
                if answer[need_list[0]] == 0 and answer[need_list[1]] == 2:
                    a0_2+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac0_2 += 1
                if answer[need_list[0]] == 0 and answer[need_list[1]] == 1:
                    a0_1+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac0_1 += 1
                if answer[need_list[0]] == 0 and answer[need_list[1]] == 0:
                    a0_0+= 1
                    if answer[need_list[0]] == int(result[need_list[0]]) and answer[need_list[1]] == int(result[need_list[1]]):
                        ac0_0 += 1
    print(fname+'の分析結果(組み合わせ)')
    print('(2,2):'+str(a2_2)+',正解数：',ac2_2)
    print('(2,0):'+str(a2_0)+',正解数：',ac2_0)
    print('(2,1):'+str(a2_1)+',正解数：',ac2_1)
    print('(0,2):'+str(a0_2)+',正解数：',ac0_2)
    print('(0,0):'+str(a0_0)+',正解数：',ac0_0)
    print('(0,1):'+str(a0_1)+',正解数：',ac0_1)
    print('(1,2):'+str(a1_2)+',正解数：',ac1_2)
    print('(1,0):'+str(a1_0)+',正解数：',ac1_0)
    print('(1,1):'+str(a1_1)+',正解数：',ac1_1)
    
            



    c0_1,c0_2,c1_0,c1_2,c2_0,c2_1,c0_0,c1_1,c2_2 = 0,0,0,0,0,0,0,0,0
    for i in range(0,len(aspect)):
        if result[i] == '0' and answer[i] == 1 :
            c0_1 += 1          
        if result[i] == '0' and answer[i] == 2 :
            c0_2 += 1
        if result[i] == '1' and answer[i] == 0 :
            c1_0 += 1
        if result[i] == '1' and answer[i] == 2 :
            c1_2 += 1
        if result[i] == '2' and answer[i] == 0 :
            c2_0 += 1
        if result[i] == '2' and answer[i] == 1 :
            c2_1 += 1
        if result[i] == '2' and answer[i] == 2:
            c2_2 += 1
        if result[i] == '1' and answer[i] == 1:
            c1_1 += 1
        if result[i] == '0' and answer[i] == 0 :
            c0_0 += 1
    print(fname+'の分析結果(aspect数:2以上)')
    print('(result,answer) = (2,2):'+str(c2_2))
    print('(result,answer) = (2,0):'+str(c2_0))
    print('(result,answer) = (2,1):'+str(c2_1))
    print('(result,answer) = (0,2):'+str(c0_2))
    print('(result,answer) = (0,0):'+str(c0_0))
    print('(result,answer) = (0,1):'+str(c0_1))
    print('(result,answer) = (1,2):'+str(c1_2))
    print('(result,answer) = (1,0):'+str(c1_0))
    print('(result,answer) = (1,1):'+str(c1_1))
    '''

    '''
    with open(fname+'_test_attention.txt', mode = 'w') as f:
        for i in range(0,len(aspect)):
            if result[i] != answer[i]:
                write_text = '[result:'+str(result[i])+', answer:'+str(answer[i])+', aspect:'+aspect[i] + ']\n'+text[i]
                f.write(write_text + '\n')
    '''
