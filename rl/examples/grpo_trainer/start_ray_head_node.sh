LARGE_N=99999999999
export RAY_health_check_initial_delay_ms=$LARGE_N
export RAY_health_check_period_ms=$LARGE_N
export NCCL_SOCKET_IFNAME=eth0

ray start --head --dashboard-host=0.0.0.0