import os


def list_dir_with_identification(
        dirname,
        identification=None,
        position='end',
        full_path=False):

    dir_content_list = os.listdir(dirname)
    if identification:
        if position == 'end':
            dir_content_list = [
                _ for _ in dir_content_list if _.endswith(identification)]
        elif position == 'in':
            dir_content_list = [
                _ for _ in dir_content_list if identification in _]
        else:
            raise NameError('parameter position is illegal')
    if not full_path:
        return dir_content_list
    else:
        return [os.path.join(dirname, _) for _ in dir_content_list]
