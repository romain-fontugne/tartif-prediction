import requests

url = "http://ihr.iijlab.net/ihr/api/delay/?asn=7922&timebin__gte=2017-05-01T00:00&timebin__lte=2017-11-01T00:00"

while url is not None:
    resp = requests.get(url)
    if (resp.ok):
        data = resp.json()
        for res in data["results"]:
            print(",".join(
                [res["timebin"], str(res["magnitude"]), str(res["asn"])]
                ))

        if "next" in data:
            url = data["next"]
        else:
            url = None
    else:
        resp.raise_for_status()
