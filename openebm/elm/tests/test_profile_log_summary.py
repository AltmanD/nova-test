from pathlib import Path

from openebm.elm.scripts.profile_log_summary import (
    build_markdown,
    load_steps,
    split_stages,
    summarize_stage,
)


def test_profile_log_summary_parses_step_families_and_stage_delta(tmp_path: Path):
    log_path = tmp_path / "profile.log"
    log_path.write_text(
        "\n".join(
            [
                "[profile_step] step=00001/3 progress=33.33% loss=1.0 tok_s=1,024 step_ms=100.0 mb=32",
                "[profile_timing] step=00001 wait_avg_ms=2.0 wait_max_ms=4.0 fw_avg_ms=20.0 bw_avg_ms=70.0 opt_ms=8.0",
                "[profile_data] step=00001 dl_avg_ms=3.0 dl_max_ms=5.0 fetch_avg_ms=1.0 tok_avg_ms=1.5 tok_max_ms=2.0 pack_avg_ms=0.5 pack_max_ms=0.9 copy_avg_ms=0.1",
                "[profile_state] step=00001 pq=0 rg=8 doc_batch=2 docbuf_end=100 docbuf_max=1000 refill_calls=4 packed_docs=64 cropped_docs=2 packed_tok=2048 cropped_tok=64",
                "[profile_step] step=00002/3 progress=66.67% loss=0.9 tok_s=512 step_ms=200.0 mb=32",
                "[profile_timing] step=00002 wait_avg_ms=20.0 wait_max_ms=50.0 fw_avg_ms=25.0 bw_avg_ms=150.0 opt_ms=10.0",
                "[profile_data] step=00002 dl_avg_ms=30.0 dl_max_ms=80.0 fetch_avg_ms=4.0 tok_avg_ms=20.0 tok_max_ms=60.0 pack_avg_ms=5.0 pack_max_ms=10.0 copy_avg_ms=0.2",
                "[profile_state] step=00002 pq=1 rg=16 doc_batch=4 docbuf_end=200 docbuf_max=1000 refill_calls=5 packed_docs=50 cropped_docs=8 packed_tok=1900 cropped_tok=320",
                "[profile_step] step=00003/3 progress=100.00% loss=0.8 tok_s=256 step_ms=400.0 mb=32",
                "[profile_timing] step=00003 wait_avg_ms=40.0 wait_max_ms=90.0 fw_avg_ms=30.0 bw_avg_ms=300.0 opt_ms=12.0",
                "[profile_data] step=00003 dl_avg_ms=60.0 dl_max_ms=120.0 fetch_avg_ms=8.0 tok_avg_ms=40.0 tok_max_ms=90.0 pack_avg_ms=10.0 pack_max_ms=20.0 copy_avg_ms=0.3",
                "[profile_state] step=00003 pq=2 rg=24 doc_batch=6 docbuf_end=300 docbuf_max=1000 refill_calls=6 packed_docs=40 cropped_docs=16 packed_tok=1800 cropped_tok=640",
            ]
        ),
        encoding="utf-8",
    )

    steps = load_steps(log_path)

    assert [step["step"] for step in steps] == [1, 2, 3]
    assert steps[0]["max_steps"] == 3
    assert steps[0]["tok_s"] == 1024
    assert steps[-1]["pq"] == 2

    stages = [summarize_stage(label, chunk) for label, chunk in split_stages(steps)]
    markdown = build_markdown("profile-test", log_path, steps, stages)

    assert stages[0]["tok_s_avg"] == 1024
    assert stages[-1]["step_ms_avg"] == 400.0
    assert "| tok_s | 1024 | 256 | -768 |" in markdown


def test_profile_log_summary_includes_dataloader_bottleneck_fields(tmp_path: Path):
    log_path = tmp_path / "profile_fine.log"
    log_path.write_text(
        "\n".join(
            [
                "[profile_step] step=00001/2 progress=50.00% loss=1.0 tok_s=1000 step_ms=100.0 mb=2",
                "[profile_timing] step=00001 wait_avg_ms=10.0 wait_max_ms=20.0 fw_avg_ms=30.0 bw_avg_ms=40.0 opt_ms=5.0",
                "[profile_data] step=00001 dl_avg_ms=20.0 dl_max_ms=24.0 fetch_avg_ms=1.0 tok_avg_ms=2.0 tok_max_ms=3.0 pack_avg_ms=4.0 pack_max_ms=5.0 doc_select_avg_ms=1.5 doc_select_max_ms=2.0 crop_branch_avg_ms=0.5 crop_branch_max_ms=1.0 tensor_avg_ms=3.0 tensor_max_ms=4.0 row_write_avg_ms=2.0 row_write_max_ms=3.0 cpu_batch_copy_avg_ms=0.7 cpu_batch_copy_max_ms=0.9 state_dict_avg_ms=5.0 state_dict_max_ms=6.0 unaccounted_avg_ms=2.0 unaccounted_max_ms=3.0 copy_avg_ms=0.2",
                "[profile_state] step=00001 pq=0 rg=8 doc_batch=1 docbuf_start_avg=1000 docbuf_end=1001 docbuf_max=1005 refill_calls=1 selected_docs=64 packed_docs=63 cropped_docs=1 packed_tok=65000 cropped_tok=500 crop_ratio=0.01 doc_len_mean=1015.0 doc_len_p50=1000.0 doc_len_p90=1500.0 doc_len_p99=2000.0 doc_len_max=2200 remaining_mean=500.0 remaining_min=500 remaining_max=500",
                "[profile_step] step=00002/2 progress=100.00% loss=0.9 tok_s=800 step_ms=125.0 mb=2",
                "[profile_timing] step=00002 wait_avg_ms=25.0 wait_max_ms=50.0 fw_avg_ms=30.0 bw_avg_ms=40.0 opt_ms=5.0",
                "[profile_data] step=00002 dl_avg_ms=60.0 dl_max_ms=70.0 fetch_avg_ms=1.0 tok_avg_ms=2.0 tok_max_ms=3.0 pack_avg_ms=4.0 pack_max_ms=5.0 doc_select_avg_ms=6.0 doc_select_max_ms=8.0 crop_branch_avg_ms=12.0 crop_branch_max_ms=16.0 tensor_avg_ms=20.0 tensor_max_ms=24.0 row_write_avg_ms=18.0 row_write_max_ms=22.0 cpu_batch_copy_avg_ms=1.3 cpu_batch_copy_max_ms=1.8 state_dict_avg_ms=19.0 state_dict_max_ms=21.0 unaccounted_avg_ms=8.0 unaccounted_max_ms=10.0 copy_avg_ms=0.3",
                "[profile_state] step=00002 pq=1 rg=16 doc_batch=2 docbuf_start_avg=1002 docbuf_end=1004 docbuf_max=1010 refill_calls=2 selected_docs=40 packed_docs=10 cropped_docs=30 packed_tok=40000 cropped_tok=25000 crop_ratio=0.38 doc_len_mean=1800.0 doc_len_p50=1700.0 doc_len_p90=2600.0 doc_len_p99=3200.0 doc_len_max=4096 remaining_mean=900.0 remaining_min=64 remaining_max=2048",
            ]
        ),
        encoding="utf-8",
    )

    steps = load_steps(log_path)
    stages = [summarize_stage(label, chunk) for label, chunk in split_stages(steps)]
    markdown = build_markdown("profile-fine", log_path, steps, stages)

    assert stages[-1]["tensor_avg_ms_avg"] == 20.0
    assert stages[-1]["row_write_avg_ms_avg"] == 18.0
    assert stages[-1]["crop_ratio_avg"] == 0.38
    assert stages[-1]["doc_len_p99_avg"] == 3200.0
    assert stages[-1]["state_dict_avg_ms_avg"] == 19.0
    assert stages[-1]["unaccounted_avg_ms_avg"] == 8.0
    assert "## Dataloader Bottleneck Summary" in markdown
    assert "| state_dict_avg_ms | 5.00 | 19.00 | 14.00 |" in markdown
    assert "| unaccounted_avg_ms | 2.00 | 8.00 | 6.00 |" in markdown
    assert "| tensor_avg_ms | 3.00 | 20.00 | 17.00 |" in markdown
    assert "| crop_ratio | 0.01 | 0.38 | 0.37 |" in markdown
