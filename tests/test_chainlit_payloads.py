"""Regression checks for fields Chainlit's frontend treats as non-null."""

import chainlit as cl


def test_download_files_always_have_mime_types(tmp_path):
    h5ad = tmp_path / "annotated.h5ad"
    counts = tmp_path / "counts.csv"
    h5ad.touch()
    counts.touch()

    elements = [
        cl.File(
            thread_id="test", name=h5ad.name, path=str(h5ad), mime="application/x-hdf5"
        ),
        cl.File(
            thread_id="test", name=counts.name, path=str(counts), mime="text/csv"
        ),
    ]

    assert all(element.mime is not None for element in elements)
    assert elements[0].mime == "application/x-hdf5"
    assert elements[1].mime == "text/csv"
