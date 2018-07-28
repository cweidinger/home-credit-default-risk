import glob
import pandas as pd
import re
from os.path import basename


def find_matches_in_list(matcher, strings):
    return [c for c in strings if re.match(matcher, c)]


class Datasets:
    def __init__(self, path="*.csv", exclude_paths=None, target="TARGET", id_matcher="SK.*ID.*"):
        self.exclude_paths = exclude_paths
        self.path = path
        self.target = target
        self.id_matcher = id_matcher
        # match files
        # MAYBE: exclude files with only TARGET and ID ... submission files
        files_to_exclude = []
        if self.exclude_paths:
            for path_to_exclude in self.exclude_paths:
                files_to_exclude = files_to_exclude + glob.glob(path_to_exclude)
        self.files = [file for file in glob.glob(path) if file not in files_to_exclude]
        self.dfs = []
        self.used = []
        self.tree = self._discover_datasets_tree()

    def accumulate_column_statistics(self, column_description_filename):
        cs = pd.read_csv(column_description_filename, nrows=200)
        for file in self.files:
            df = pd.read_csv(file)  # load whole file to get statistics
            filename = basename(file)
            sz = len(df.index)
            for col in df.columns:
                indexes = cs[(cs['filename'] == filename) & (cs['column'] == col)].index
                if len(indexes) == 1:
                    idx = indexes[0]
                    cs.set_value(idx, 'dtype', df[col].dtype)
                    cs.set_value(idx, 'unique_count', len(df[col].unique()))
                    cs.set_value(idx, 'nan_percent', df[col].isna().sum() / sz * 100)
                else:
                    print("{}-{} exists in the Input but couldn't be found in the column descriptions file. Perhaps it's misnamed.".format(filename, col))
        cs.to_csv(column_description_filename, index=False)

    def find_children(self, base_index, base_id, level):
        kids = []
        for index, file in enumerate(self.files):
            if index == base_index or self.used[index]:
                continue
            ids = find_matches_in_list(self.id_matcher, self.dfs[index].columns)
            if base_id in ids:
                self.used[index] = True
                ids.remove(base_id)
                kids.append((index, ids))
        children = []
        for kid in kids:
            index = kid[0]
            ids = kid[1]
            children.append({
                "id": base_id,
                "filename": self.files[index],
                "index": index,
                "children": self.find_children(index, ids[0], level + 1) if len(ids) != 0 else [],
                "columns": self.dfs[index].columns
            })
        return children

    def _discover_datasets_tree(self):
        # Load dfs and find target
        target_df_index = None
        for index, file in enumerate(self.files):
            df = pd.read_csv(file, nrows=200)
            self.dfs.append(df)
            if self.target in df.columns:
                if target_df_index:
                    raise Exception("2 {}'s found: {} and {}".format(self.target, self.files[target_df_index], self.files[index]))
                target_df_index = index
                self.used.append(True)
            else:
                self.used.append(False)
        ids = find_matches_in_list(self.id_matcher, self.dfs[target_df_index].columns)
        if len(ids) != 1:
            raise Exception("Need exactly 1 id matching {} in file with '{}' but found {} in {}".format(
                self.id_matcher,
                self.target,
                ids,
                self.files[target_df_index])
            )
        print("'{}' found in '{}' with id '{}'".format(self.target, self.files[target_df_index], ids[0]))
        # build tree
        tree = {
            "id": ids[0],
            "filename": self.files[target_df_index],
            "target": self.target,
            "index": target_df_index,
            "children": self.find_children(target_df_index, ids[0], 0),
            "columns": self.dfs[target_df_index].columns
        }
        # MAYBE: exclude files with no link to TARGET OR just throw up
        for index, u in enumerate(self.used):
            if not u:
                raise Exception("{} matched but was not related to '{}'".format(
                    self.files[index],
                    self.target)
                )
        return tree
