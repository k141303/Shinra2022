import os

from collections import Counter

from utils.data_utils import DataUtils


class EneData(object):
    def __init__(self, file_path):
        self.pageid2ene_ids, self.ene_id_list = self._load(file_path)

    @staticmethod
    def _load(file_path):
        data = DataUtils.JsonL.load(file_path)
        pageid2ene_ids = {}
        label_cnt = Counter()
        for d in data:
            assert len(d["ENEs"]) == 1, "ENEs key error."

            ene_ids = []
            for ene in list(d["ENEs"].values())[0]:
                try:
                    ene_id = ene["ENE"]
                except KeyError:
                    ene_id = ene["ENE_id"]
                if len(ene_id) == 0:
                    continue
                ene_ids.append(ene_id)
                label_cnt[ene_id] += 1

            if len(ene_ids) == 0:
                continue

            try:
                pageid2ene_ids[str(d["page_id"])] = ene_ids
            except KeyError:
                pageid2ene_ids[str(d["pageid"])] = ene_ids
        ene_id_list = sorted(label_cnt.keys())
        return pageid2ene_ids, ene_id_list

    def get_ene_ids(self, pageid):
        return self.pageid2ene_ids.get(str(pageid))

    def get_pageids(self):
        return set(self.pageid2ene_ids.keys())

    def save_ene_id_list(self, output_dir):
        DataUtils.Json.save(os.path.join(output_dir, "ene_id_list.json"), self.ene_id_list)
