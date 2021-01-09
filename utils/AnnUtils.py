import re

import pandas as pd
import csv


def get_primative_abbreviation(char):
    if char == 'I':
        return 'int'
    elif char == 'Z':
        return 'long'
    elif char == 'F':
        return 'float'
    elif char == 'J':
        return 'boolean'
    elif char == 'D':
        return 'double'
    elif char == 'C':
        return 'char'
    elif char == 'S':
        return 'short'
    elif char == 'V':
        return 'void'
    elif char == 'B':
        return 'byte'
    else:
        return None


def convert_parameters(parameters):
    parsed_parameters = []
    is_object = False
    curr_p_start = 0
    for i in range(len(parameters)):
        if parameters[i] == ';':
            is_object = False
            object = parameters[curr_p_start:i]
            parsed_parameters.append(get_class_name(object))

        elif is_object or parameters[i] == '[':
            continue

        elif parameters[i] == 'L':
            is_object = True
            curr_p_start = i+1

        else:
            prim = get_primative_abbreviation(parameters[i])
            if prim is not None:
                parsed_parameters.append(prim)

    return parsed_parameters


def get_class_name(java_name):
    return re.sub(r".*\.(.*?)$", r"\1", java_name)


def convert_ret_type(ret_type):
    ret_type = ret_type.strip(";[")
    if len(ret_type) > 1 and ret_type[0] == 'L':
        ret_type = get_class_name(ret_type)
    elif len(ret_type) == 1:
        ret_type = get_primative_abbreviation(ret_type)
    else:
        print("PROBLEM WITH RET TYPE:", ret_type)

    return ret_type


def adjoin_sig_pieces(qual_class_name, method_name, parameters, ret_type):
    ret = "<"+qual_class_name.strip()+": "+ret_type.strip()+" "+method_name.strip()
    return ret + '(' + ",".join(parameters)+")>"


def make_method_id(row):
    if len(row[0]) >= 3 and row[0][:7] == 'android':
        for i in range(3):
            row[i] = re.sub("/", ".", row[i])
        (qual_classname, method_name, descriptors) = row[:3]
        descriptors = descriptors.lstrip("( ")
        (parameters, ret_type) = descriptors.split(")")
        ret_type = convert_ret_type(ret_type)
        parameters = convert_parameters(parameters)
        return adjoin_sig_pieces(qual_classname, method_name, parameters, ret_type)
    else:
        return None


def make_annotation_subset(df:pd.DataFrame, k:int, cols_to_use:list, new_cols_for_anns:list, do_not_use:pd.Series=None):
    df = df.loc[~df.index.duplicated(keep='first')]
    if k > df.shape[0] :
        print("WARNING: subset is larger than total set", k,">", df.shape[0])
        print("\tSetting subset equal to", df.shape[0])
        k = df.shape[0]
    df = df.sample(n=k)[cols_to_use]
    for col in new_cols_for_anns:
        df[col] = ''
    return df


def get_pscout_methods(filename:str):
    with open(filename, newline='', encoding='utf8') as f:
        reader = csv.reader(f, dialect='excel')
        api_methods = []
        for row in reader:
            method_id = make_method_id(row)
            if method_id is not None:
                api_methods.append(method_id)
    return pd.Series(api_methods)


def get_df_from_csv(filename:str, aggregate_cols:tuple=None, col_names:list=None
                    , ignore_if_next_contains:list=None, index_col=None, add_period=False):
    df = pd.read_csv(filename,header=None)
    temp = pd.Series(['' for i in range(df.shape[0])])

    for i in range(aggregate_cols[0], aggregate_cols[1]+1):
        temp = temp + df[i]
        if i < aggregate_cols[1]:
            for pat in ignore_if_next_contains:
                temp.loc[df[i+1].str.contains(pat)] = ''
            temp = temp + '.'

    if not add_period:
        temp = temp.str.strip('.')

    ret_df = df[[i for i in range(aggregate_cols[0])]]
    if not ret_df.empty:
        ret_df = pd.concat([ret_df,temp], axis=1)
    else:
        ret_df = temp

    if aggregate_cols[1] < df.shape[1]:
        ret_df = pd.concat([ret_df,df[[i for i in range(aggregate_cols[1]+1, df.shape[1])]]], axis=1)

    ret_df.columns = col_names
    if index_col is not None and index_col in col_names:
        ret_df.index = ret_df[index_col]
        ret_df = ret_df.drop(index_col, axis=1)


    return ret_df


def add_extra_descrips(df:pd.DataFrame, descrips:pd.DataFrame, descrip_col_name=''):
    df_idx = df.index.to_series()
    df[descrip_col_name] = pd.Series(['' for i in range(df.shape[0])])
    for signature_piece in descrips.index.to_list():
        # TODO: There is a bug in this loop. Should match longest package name. Add if statement to fix.
        # TODO: (i.e. if xxxx is not NA check something)
        df.loc[df_idx.str.contains(signature_piece, regex=False), descrip_col_name] = descrips.loc[signature_piece,'Description']
    df[descrip_col_name] = df[descrip_col_name].fillna('')
    return df

def create_and_save_annotations(df, num_in_subset, cols_to_use, new_cols, class_descrips, package_descrips,
                                save_filename, do_not_use=None):
    '''


    :param df:
    :param num_in_subset:
    :param cols_to_use:
    :param new_cols:
    :return:
    '''
    df = make_annotation_subset(df.loc[df["Classname"].notna()], num_in_subset, cols_to_use,
                                new_cols, do_not_use=do_not_use)

    df = add_extra_descrips(df, class_descrips, descrip_col_name='ClassDescription')
    df = add_extra_descrips(df, package_descrips, descrip_col_name='PackageDescription')
    df = df[cols_to_use + ['ClassDescription', 'PackageDescription'] + new_cols]
    df.to_csv(save_filename)


def create_annotations(df, num_in_subset, cols_to_use, new_cols, class_descrips, package_descrips,
                                save_filename):
    '''


    :param df:
    :param num_in_subset:
    :param cols_to_use:
    :param new_cols:
    :return:
    '''
    df = make_annotation_subset(df.loc[df["Classname"].notna()], num_in_subset, cols_to_use,
                                new_cols)

    df = add_extra_descrips(df, class_descrips, descrip_col_name='ClassDescription')
    df = add_extra_descrips(df, package_descrips, descrip_col_name='PackageDescription')
    df = df[cols_to_use + ['ClassDescription', 'PackageDescription'] + new_cols]
    return df
