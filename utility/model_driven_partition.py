def model_driven_partition(config_json):
    y_axis = []
    split = []
    model_tuning = config_json['tuning']

    b_cache = config_json['b_cache']
    s_cache = config_json['s_cache']
    ib_sc = config_json['b_storage_nw']
    b_storage = config_json['b_storage']

    t_a = config_json['t_a']
    t_da = config_json['t_da']
    t_gpu = config_json['t_gpu']

    # Confirm these are lists; if not, wrap them in a list
    ib_ct = config_json['network'] if isinstance(config_json['network'], (list, tuple)) else [config_json['network']]
    ib_cp = config_json['network'] if isinstance(config_json['network'], (list, tuple)) else [config_json['network']]
    ib_ca = config_json['network'] if isinstance(config_json['network'], (list, tuple)) else [config_json['network']]

    ib_pt = config_json['pcie'] if isinstance(config_json['pcie'], (list, tuple)) else [config_json['pcie']]
    ib_at = config_json['pcie'] if isinstance(config_json['pcie'], (list, tuple)) else [config_json['pcie']]

    n_total = int(config_json['ds']/config_json['s_data'])
    s_data = config_json['s_data']
    s_data_mem = config_json['s_data_mem']
    m_s_data = config_json['s_data']*config_json['m']


    for i in range(0,100,1):
        for j in range(0,100-i,1):
            k=100-(i+j)
            h=[i/100,j/100,k/100]

            remaining_tensors = n_total

            tensor_ppd = min(remaining_tensors, ((h[2] * s_cache) / (2 * m_s_data)))
            remaining_tensors -= tensor_ppd
            tensor_decoded = min(remaining_tensors, ((h[1] * s_cache) / m_s_data))
            remaining_tensors -= tensor_decoded
            tensor_rd = min(remaining_tensors, (((h[0]) * s_cache) / s_data_mem))
            remaining_tensors -= tensor_rd

            remaining_tensors = max(remaining_tensors, 0)
            tensor_storage = remaining_tensors

            b_perf_dist = (
                        ((tensor_ppd / n_total) * min((b_cache / m_s_data), (sum(ib_ct) / (m_s_data)), (sum(t_gpu)))) +
                        ((tensor_decoded / n_total) * min((b_cache / m_s_data), (sum(ib_ca) / (m_s_data)), (sum(t_a)),
                                                          (sum(ib_at) / m_s_data), (sum(t_gpu)))) +
                        ((tensor_rd / n_total) * min((b_cache / s_data_mem), (sum(ib_cp) / (s_data_mem)), (sum(t_da)),
                                                     (sum(ib_pt) / m_s_data), (sum(t_gpu)))) +
                        ((tensor_storage / n_total) * min((b_storage / s_data), (ib_sc / s_data),
                                                          (b_cache / s_data_mem), (sum(ib_cp) / (s_data_mem)),
                                                          (sum(t_da)), (sum(ib_pt) / m_s_data), (sum(t_gpu))))
                        )
            y_axis.append((b_perf_dist * model_tuning) * 1)
            split.append(h)
    split_index = y_axis.index(max(y_axis))
    return (split[split_index])

