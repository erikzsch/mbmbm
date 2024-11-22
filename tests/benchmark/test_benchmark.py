# def test_benchmark(test_resources, config_dir, tmp_path):
#     configs_dir = str(config_dir / "benchmark")
#     os.environ["MBMBM_DATASET_DIR"] = str(test_resources)
#
#     bench = Benchmark(workdir=tmp_path, configs_dir=configs_dir)
#     bench.run_trains()
#     bench.run_vals()
#
#     bench_path = tmp_path / "benchmark_results"
#     model_path_list = [
#         bench_path / "sklearn_basic",
#         bench_path / "sklearn_basic_neural",
#     ]
#
#     def check_res(model_path):
#         pred = list(model_path.glob("prediction*"))
#         tar = list(model_path.glob("targets*"))
#         assert len(pred) == 1, f"{len(pred)=}!=1"
#         assert len(tar) == 1, f"{len(pred)=}!=1"
#
#         arr_pred = np.load(pred[0])
#         arr_tar = np.load(tar[0])
#
#         assert len(arr_pred) == 113
#         assert arr_tar.shape == (113,)
#
#     for mp in model_path_list:
#         check_res(mp)
#     print("Done.")
