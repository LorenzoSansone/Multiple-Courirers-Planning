import multiprocessing as mp

if __name__ == "__main__":
    with mp.Manager() as manager:
        shared_res = manager.dict()

        shared_res["res"] = [300, False,None,None]
        shared_res["res"][2:] = [[["obj_value"]], 15]
        print(shared_res)

