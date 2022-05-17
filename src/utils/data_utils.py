import os
import re
import glob

import json
import gzip

import tqdm

from multiprocessing import Pool
import multiprocessing as multi

from collections import defaultdict


class DataUtils:
    class File(object):
        def load(file_path):
            with open(file_path, "r") as f:
                return f.read()

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

    class Json(object):
        def load(file_path):
            with open(file_path, "r") as f:
                return json.load(f)

        def save(file_path, data):
            with open(file_path, "w") as f:
                json.dump(
                    data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": ")
                )

    class CirrusSearch(object):
        def load(file_path, pageids=None, return_keys=["text"], debug_mode=False):
            data = []
            with gzip.open(file_path, mode="rt") as f, tqdm.tqdm(desc=f"Load CirrusSearch") as t:
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

    class AttrExtData(object):
        def mp_load_plain_texts(file_path):
            data = DataUtils.File.load(file_path)
            lines = data.splitlines()

            for i in range(min(10, len(lines))):
                lines[i] = ""
            for i in range(min(5, len(lines))):
                lines[-i] = ""

            pageid = os.path.splitext(os.path.basename(file_path))[0]
            category = file_path.split(os.path.sep)[-2]

            return category, pageid, "\n".join(lines)

        @classmethod
        def load_plain_texts(cls, file_dir):
            file_paths = glob.glob(os.path.join(file_dir, f"plain/*/*.txt"))

            data = {}
            with Pool(multi.cpu_count()) as p, tqdm.tqdm(
                desc="Loading plain texts", total=len(file_paths)
            ) as t:
                for category, pageid, d in p.imap(cls.mp_load_plain_texts, file_paths):
                    data[(category, pageid)] = d
                    t.update()

            return data

        def ml_load_annotation_data(file_path):
            data = DataUtils.JsonL.load(file_path)
            category = re.match(".*/(.*?)_dist.jsonl", file_path).group(1)
            return category, data

        @classmethod
        def load_annotation_data(cls, file_dir, plain_texts):
            file_paths = glob.glob(os.path.join(file_dir, f"annotation/*_dist.jsonl"))

            all_data, attributes = {}, defaultdict(set)
            with Pool(multi.cpu_count()) as p, tqdm.tqdm(
                desc="Loading annotation_data", total=len(file_paths)
            ) as t:
                for category, data in p.imap(cls.ml_load_annotation_data, file_paths):
                    for d in data:
                        pageid = d["pageid"] if "pageid" in d else d["page_id"]
                        d["text"] = plain_texts[(category, pageid)]
                        attributes[category].add(d["attribute"])
                    all_data[category] = data
                    t.update()

            return all_data, {k: sorted(v) for k, v in attributes.items()}

        @classmethod
        def load(cls, file_dir):
            plain_texts = cls.load_plain_texts(file_dir)
            annotation_data, attributes = cls.load_annotation_data(file_dir, plain_texts)

            return annotation_data, attributes
