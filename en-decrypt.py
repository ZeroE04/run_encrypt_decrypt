from typing import List
from typing import Tuple
from simplediff import diff
import numpy as np


class run_encrypt_decrypt:
    def __init__(self):
        self.class_mapping = {'none': 0, 'replace': 1, 'before': 2}
        
    def encrypt(self, context: List,current_sen: str,label_sen: str,
                super_mode: str = 'before',only_one_insert: bool = False,
                special_token: bool = True):
        
        context_seq = []
        for txt in context:
            txt_seq = list(txt)
            if special_token:
                txt_seq+=['[SEP]']
            context_seq.extend(txt_seq)
        current_seq, label_seq = list(current_sen),list(label_sen)
        if special_token:
            current_seq+=['[END]']
        applied_changes = diff(current_seq, label_seq)
    
        def sub_finder(cus_list, pattern, used_pos):
            find_indices = []
            for i in range(len(cus_list)):
                if cus_list[i] == pattern[0] and \
                        cus_list[i:i + len(pattern)] == pattern \
                        and i not in used_pos:
                    find_indices.append((i, i + len(pattern)))
            if len(find_indices) == 0:
                return 0, 0
            else:
                return find_indices[-1]
    
        def cont_sub_finder(cus_list, pattern, used_pos):
            context_len = len(cus_list)
            pattern_len = len(pattern)
            for i in range(context_len):
                k = i
                j = 0
                temp_indices = []
                while j < pattern_len and k < context_len:
                    if cus_list[k] == pattern[j][0] and \
                            cus_list[k:k + len(pattern[j])] == pattern[j] \
                            and k not in used_pos:
                        temp_indices.append((k, k + len(pattern[j])))
                        j += 1
                    else:
                        k += 1
                if j == pattern_len:
                    return zip(*temp_indices)
            else:
                return 0, 0
    
        rm_range = None
        ret_ops = []
        context_used_pos = []
        current_used_pos = []
        pointer = 0
        for diff_sample in applied_changes:
            diff_op = diff_sample[0]
            diff_content = diff_sample[1]
            if diff_op == '-':
                if rm_range is not None:
                    ret_ops.append(['remove', rm_range, []])
                start, end = sub_finder(current_seq, diff_content, current_used_pos
                                        )
                rm_range = [start, end]
                current_used_pos.extend(list(range(start, end)))
            elif diff_op == '+':
                start, end = sub_finder(context_seq, diff_content, context_used_pos)
                # cannot find the exact match substring, we should identify the snippets
                if start == 0 and end == 0:
                    inner_diff = diff(diff_content, context_seq)
                    overlap_content = [inner_diff_sample[1] for
                                       inner_diff_sample in inner_diff if inner_diff_sample[0] == '=']
                    if len(overlap_content) > 0:
                        # only take one insert
                        if len(overlap_content) == 1 or only_one_insert:
                            overlap_content = sorted(overlap_content, key=lambda x: len(x), reverse=True)[0]
                            start, end = sub_finder(context_seq, overlap_content,
                                                    context_used_pos)
                        else:
                            start_end_tuple = cont_sub_finder(context_seq, overlap_content, context_used_pos)
                            # start is a list, end is also
                            start, end = start_end_tuple
                    else:
                        start, end = 0, 0
                if not (start == 0 and end == 0):
                    if isinstance(start, int):
                        add_ranges = [[start, end]]
                    else:
                        add_ranges = list(zip(start, end))
    
                    if rm_range is not None:
                        for add_range in add_ranges:
                            context_used_pos.extend(list(range(add_range[0], add_range[1])))
                            ret_ops.append(['replace', rm_range, add_range])
                        rm_range = None
                    else:
                        for add_range in add_ranges:
                            if super_mode in ['before', 'both']:
                                ret_ops.append(['before', [pointer, pointer], add_range])
                            if super_mode in ['after', 'both']:
                                if pointer >= 1:
                                    ret_ops.append(['after', [pointer - 1, pointer - 1], add_range])
            elif diff_op == '=':
                if rm_range is not None:
                    ret_ops.append(['remove', rm_range, []])
                start, end = sub_finder(current_seq, diff_content, current_used_pos
                                        )
                current_used_pos.extend(list(range(start, end)))
                rm_range = None
                pointer = end
            matrix_map = np.zeros((len(context_seq), len(current_seq)),dtype=np.long)
            for op_tuple in ret_ops:
                op_name = op_tuple[0]
                label_value = self.class_mapping[op_name]
                cur_start, cur_end = op_tuple[1]
                con_start, con_end = op_tuple[2]
                
                if op_name == 'replace':
                    matrix_map[con_start:con_end, cur_start:cur_end] = label_value
                else:
                    assert cur_start == cur_end
                    matrix_map[con_start:con_end, cur_start] = label_value
        return matrix_map, context_seq, current_seq

    def decrypt(self, attn_map, cur_str, context_str,attn_mask=None) -> str:
        """
        Detection the operation op, keeping the same format as the result of export_conflict_map
        :param attn_map: attention_map, with shape `height x width x class_size`
        :return: ordered operation sequence
        """
    
        discrete_attn_map = attn_map
        #    discrete_attn_map = np.argmax(attn_map, axis=2)
        #    discrete_attn_map = attn_mask * discrete_attn_map
        op_seq: List = []
    
        for label, label_value in self.class_mapping.items():
            if label_value == 0:
                # do nothing
                continue
            connect_matrix = discrete_attn_map.copy()
            
            # make the non label value as zero
            connect_matrix = np.where(connect_matrix != label_value, 0,
                                      connect_matrix)
            
            ops = self._scan_twice(connect_matrix)
            for op in ops:
                op_seq.append([label, *op])
        
        op_seq = sorted(op_seq, key=lambda x: x[2][1], reverse=True)
        predict_str = self._transmit_seq(cur_str, context_str, op_seq)
        return predict_str

    def _transmit_seq(self, cur_str: str, context_str: str,
                     op_seq: List[Tuple[str, Tuple, Tuple]]) -> str:
        """
        Given an operation sequence as `add/replace`, context_start_end, cur_start_end, transmit the generated sequence
        :param op_seq:
        :return:
        """
        current_seq = cur_str.split(' ')
        context_seq = context_str.split(' ')
    
        for operation in op_seq: #[['before', [4, 5], [8, 10]], ['replace', [1, 3], [6, 8]]]
            opera_op = operation[0]
            current_range = operation[1] #[4,5]
            context_range = operation[2] #[]
            if opera_op == 'replace':
                current_seq[current_range[0]:current_range[1]] = context_seq[context_range[0]:context_range[1]]
            elif opera_op == 'before':
                current_seq[current_range[0]:current_range[0]] = context_seq[context_range[0]:context_range[1]]
            elif opera_op == 'after':
                current_seq[current_range[0] + 1: current_range[0] + 1] = context_seq[context_range[0]:context_range[1]]
    
        # remove current_seq
        ret_str = ' '.join(current_seq).strip()
        return ret_str
            
    def _scan_twice(self, connect_matrix):
        label_num = 1
        label_equations = {}
        height, width = connect_matrix.shape
        for i in range(height):
            for j in range(width):
                if connect_matrix[i, j] == 0:
                    continue
                if j != 0:
                    left_val = connect_matrix[i, j - 1]
                else:
                    left_val = 0
                if i != 0:
                    top_val = connect_matrix[i - 1, j]
                else:
                    top_val = 0
                if i != 0 and j != 0:
                    left_top_val = connect_matrix[i - 1, j - 1]
                else:
                    left_top_val = 0
                if any([left_val > 0, top_val > 0, left_top_val > 0]):
                    neighbour_labels = [v for v in [left_val, top_val,
                                                    left_top_val] if v > 0]
                    min_label = min(neighbour_labels)
                    connect_matrix[i, j] = min_label
                    set_min_label = min([label_equations[label] for label in
                                         neighbour_labels])
                    for label in neighbour_labels:
                        label_equations[label] = min(set_min_label, min_label)
                    if set_min_label > min_label:
                        for key, value in label_equations:
                            if value == set_min_label:
                                label_equations[key] = min_label
                else:
                    new_label = label_num
                    connect_matrix[i, j] = new_label
                    label_equations[new_label] = new_label
                    label_num += 1
        for i in range(height):
            for j in range(width):
                if connect_matrix[i, j] == 0:
                    continue
                label = connect_matrix[i, j]
                normalized_label = label_equations[label]
                connect_matrix[i, j] = normalized_label
        groups = list(set(label_equations.values()))
        ret_boxes = []
        for group_label in groups:
            points = np.argwhere(connect_matrix == group_label)
            points_y = points[:, (0)]
            points_x = points[:, (1)]
            min_width = np.amin(points_x)
            max_width = np.amax(points_x) + 1
            min_height = np.amin(points_y)
            max_height = np.amax(points_y) + 1
            ret_boxes.append([[min_width, max_width], [min_height, max_height]])
        return ret_boxes        
                     
if __name__=="__main__":
    context = ['需要什么', '有戴森吹风机吗', '没有']
    cur = '那松下的呢'
    restate = '那戴森的吹风机呢'
    ED = run_encrypt_decrypt()
    
    # 编码过程->根据历史对话、当前会话、标签改写会话，构造标签矩阵label
    label,tokenized_context,tokenized_cur = ED.encrypt(context,cur,restate)
    print(label)
    
    # 解码过程->根据预测矩阵、历史对话、当前对话，输出当前对话改写内容
    cur_str, context_str = ' '.join(tokenized_cur),' '.join(tokenized_context)
    result = ED.decrypt(label, cur_str, context_str)
    print(result)

    '''
    [[0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 1 1 0 0 0]
     [0 1 1 0 0 0]
     [0 0 0 0 2 0]
     [0 0 0 0 2 0]
     [0 0 0 0 2 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]
     [0 0 0 0 0 0]]
    那 戴 森 的 吹 风 机 呢 [END]
    '''
    
    




