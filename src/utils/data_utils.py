import json
import gzip

import tqdm


class DataUtils:
    class JsonL(object):
        def json_dumps(d):
            return json.dumps(d, ensure_ascii=False)

        @classmethod
        def save(cls, file_path, data):
            dumps = map(cls.json_dumps, data)
            with open(file_path, "w") as f:
                f.write("\n".join(dumps))

        def load(file_path):
            with open(file_path, "r") as f:
                return [*map(json.loads, f)]

    class CirrusSearch(object):
        def load(file_path, pageids=None, return_keys=["text"], debug_mode=False):
            data = []
            with gzip.open(file_path, mode='rt') as f, tqdm.tqdm(desc=f"Load CirrusSearch") as t:
                for d in map(json.loads, f):
                    if d.get("index", {}).get("_id") is not None:
                        pageid = str(d["index"]["_id"])
                        continue

                    if pageids is not None and pageid not in pageids:
                        continue

                    d = {key: d[key] for key in return_keys}
                    d["pageid"] = pageid
                    data.append(d)

                    t.update()

                    if debug_mode and len(data) >= 5000:
                        break
            return data
