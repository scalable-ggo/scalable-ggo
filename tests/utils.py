from hypothesis import strategies as st

from sggo.cluster import Cluster


@st.composite
def cpu_cluster(draw, max_size: int = 128) -> Cluster:
    n = draw(st.integers(min_value=0, max_value=max_size))
    return Cluster.generate(n)
