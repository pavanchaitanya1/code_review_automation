class Data:
    def __init__(self, patch_json):
        if 'patch' in patch_json:
            self.patch = patch_json['patch']
        if 'proj' in patch_json:
            self.project = patch_json['proj']
        if 'y' in patch_json:
            self.y = patch_json['y']
        if 'msg' in patch_json:
            self.msg = patch_json['msg']
        if 'score' in patch_json:
            self.score = patch_json['score']
        if 'id' in patch_json:
            self.id = patch_json['id']
        if 'patch_id' in patch_json:
            self.id = patch_json['patch_id']