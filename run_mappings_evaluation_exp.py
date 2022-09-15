
import runner


propara_mappings_eval = [('propara_para_id_490', 'propara_para_id_1158'),
                         ('propara_para_id_552', 'propara_para_id_626'),
                         ('propara_para_id_524', 'propara_para_id_645'),
                         ('propara_para_id_393', 'propara_para_id_392'),
                         ('propara_para_id_587', 'propara_para_id_588'),
                         ('propara_para_id_1291', 'propara_para_id_1014'),
                         ('propara_para_id_330', 'propara_para_id_938'),
                         ('propara_para_id_779', 'propara_para_id_938'),
                         ('propara_para_id_157', 'propara_para_id_882'),
                         ('propara_para_id_779', 'propara_para_id_330'),
                         ('propara_para_id_644', 'propara_para_id_528'),
                         ('propara_para_id_1224', 'propara_para_id_687'),
                         ('propara_para_id_1127', 'propara_para_id_7'),
                         ('propara_para_id_1158', 'propara_para_id_315')]


stories_mapping_eval = [('keane_general', 'keane_surgeon'),
           ('rattermann_story1_base', 'rattermann_story1_target'),
           ('rattermann_story2_base', 'rattermann_story2_target'),
           ('rattermann_story3_base', 'rattermann_story3_target'),
           ('rattermann_story4_base', 'rattermann_story4_target'),
           ('rattermann_story5_base', 'rattermann_story5_target'),
           ('rattermann_story8_base', 'rattermann_story8_target'),
           ('rattermann_story11_base', 'rattermann_story11_target'),
           ('rattermann_story12_base', 'rattermann_story12_target'),
           ('rattermann_story13_base', 'rattermann_story13_target'),
           ('rattermann_story14_base', 'rattermann_story14_target'),
           ('rattermann_story16_base', 'rattermann_story16_target'),
           ('rattermann_story17_base', 'rattermann_story17_target'),
           ('rattermann_story18_base', 'rattermann_story18_target')]


if __name__ == '__main__':
    model_name = runner.FMQ
    sim_threshold = runner.MODELS_SIM_THRESHOLD[model_name]
    run_coref, run_qasrl, run_mappings = False, False, True

    # FMQ on stories and proPara
    runner.run_pipeline(model_name, sim_threshold, stories_mapping_eval, run_coref, run_qasrl, run_mappings)
    runner.run_pipeline(model_name, sim_threshold, propara_mappings_eval, run_coref, run_qasrl, run_mappings)

    model_name = runner.FMV
    sim_threshold = runner.MODELS_SIM_THRESHOLD[model_name]

    # FMV on stories and proPara
    runner.run_pipeline(model_name, sim_threshold, stories_mapping_eval, run_coref, run_qasrl, run_mappings)
    runner.run_pipeline(model_name, sim_threshold, propara_mappings_eval, run_coref, run_qasrl, run_mappings)
