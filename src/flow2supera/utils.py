import os
import numpy as np
import time
import flow2supera
import ROOT
import yaml
from yaml import Loader
from larcv import larcv
from supera import supera

#from LarpixParser import event_parser as EventParser

def get_iomanager(outname):
    import tempfile
    cfg='''                                                                                                                                          
IOManager: {                                                                                                                                         
  Verbosity:   2                                                                                                                                     
  Name:        "IOManager"                                                                                                                           
  IOMode:      1                                                                                                                                     
  OutFileName: "%s"                                                                                                                                  
}                                                                                                                                                    
'''
    #f=open('tmp.cfg','w')                                                                                                                           
    f=tempfile.NamedTemporaryFile(mode='w')
    f.write(cfg % outname)
    f.flush()
    o = larcv.IOManager(f.name)
    o.initialize()
    f.close()
    return o


def larcv_meta(supera_meta):
    larcv_meta = larcv.Voxel3DMeta()

    larcv_meta.set(supera_meta.min_x(),supera_meta.min_y(),supera_meta.min_z(),
                   supera_meta.max_x(),supera_meta.max_y(),supera_meta.max_z(),
                   supera_meta.num_voxel_x(),supera_meta.num_voxel_y(),supera_meta.num_voxel_z())
    
    return larcv_meta

def larcv_particle(p):
    
    US2NS = 1.e3

    larp=larcv.Particle()
    
    larp.id              (p.part.id)
    larp.shape           (int(p.part.shape))
    
    # particle's info setter
    larp.track_id         (p.part.trackid)
    if hasattr(p.part, "genid") and hasattr(larp, "gen_id"):
        larp.gen_id(p.part.genid)
        
    larp.pdg_code         (p.part.pdg)
    larp.momentum         (p.part.px,p.part.py,p.part.pz)
    larp.end_momentum     (p.part.end_px,p.part.end_py,p.part.end_pz)
    larp.distance_travel  (p.part.dist_travel)
    
    vtx_dict = dict(position = p.part.vtx, 
                    end_position = p.part.end_pt, 
                    first_step = p.part.first_step, 
                    last_step = p.part.last_step,
                    parent_position = p.part.parent_vtx,
                    ancestor_position = p.part.ancestor_vtx,
                   )
    for key,item in vtx_dict.items():
        getattr(larp,key)(item.pos.x, item.pos.y, item.pos.z, item.time * US2NS)
    
    #larp.distance_travel ( double dist ) { _dist_travel = dist; }
    larp.energy_init      (p.part.energy_init)
    larp.energy_deposit   (p.energy.sum())
    larp.creation_process (p.part.process)
    larp.num_voxels       (p.energy.size())
    
    # parent info setter
    larp.parent_track_id (p.part.parent_trackid)
    larp.parent_pdg_code (p.part.parent_pdg)
    larp.parent_creation_process(p.part.parent_process)
    larp.parent_id       (p.part.parent_id)
    for cid in p.part.children_id:
        larp.children_id(cid)

    # ancestor info setter
    larp.ancestor_track_id (p.part.ancestor_id)
    larp.ancestor_pdg_code (p.part.ancestor_pdg)
    larp.ancestor_creation_process(p.part.ancestor_process)
                                   
    if not p.part.group_id == supera.kINVALID_INSTANCEID: 
        larp.group_id(p.part.group_id)
        
    if not p.part.interaction_id == supera.kINVALID_INSTANCEID:
        larp.interaction_id(p.part.interaction_id)
    
    return larp

def larcv_neutrino(n):
    
    larn = larcv.Neutrino()
    US2NS = 1.e3
        
    larn.id                 (larcv.InstanceID_t(n.id)) 
    larn.interaction_id     (larcv.InstanceID_t(n.interaction_id))
    larn.nu_track_id        (supera.CUInt_t(n.nu_track_id))
    larn.lepton_track_id    (supera.CUInt_t(n.lepton_track_id))
    larn.current_type        (n.current_type)
    larn.interaction_mode    (n.interaction_mode)
    larn.interaction_type    (n.interaction_type)
    larn.target              (n.target)   
    larn.nucleon             (n.nucleon)
    larn.quark               (n.quark)
    larn.hadronic_invariant_mass(n.hadronic_invariant_mass)
    larn.bjorken_x              (n.bjorken_x)
    larn.inelasticity           (n.inelasticity)
    larn.momentum_transfer      (n.momentum_transfer)
    larn.momentum_transfer_mag  (n.momentum_transfer_mag)
    larn.energy_transfer        (n.energy_transfer)
    larn.theta               (n.theta)
    larn.pdg_code            (int(n.pdg_code))
    larn.lepton_pdg_code     (int(n.lepton_pdg_code))
    larn.momentum            (n.px, n.py, n.pz)
    larn.lepton_p            (n.lepton_p)
    larn.position            (n.vtx.pos.x, n.vtx.pos.y, n.vtx.pos.z, n.vtx.time * US2NS)
    larn.distance_travel     (n.dist_travel)
    larn.energy_init         (n.energy_init)
    larn.energy_deposit      (n.energy_deposit)
    larn.creation_process    (n.creation_process)
    larn.num_voxels          (int(n.num_voxels))
   
    return larn


def get_flow2supera(config_key):

    driver = flow2supera.driver.SuperaDriver()
    if os.path.isfile(config_key):
        driver.ConfigureFromFile(config_key)
    else:
        driver.ConfigureFromFile(flow2supera.config.get_config(config_key))
    

    return driver 

def log_supera_integrity_check(data, driver, log, verbose=False):

    if not log:
        return

    label = driver.Label()
    meta  = driver.Meta()

    # Packet tensor
    pcloud = np.array([[edep.x,edep.y,edep.z,edep.e] for edep in driver._edeps_all])
    voxels = meta.edep2voxelset(driver._edeps_all)
    voxels = np.array([[meta.pos_x(vox.id()),
                        meta.pos_y(vox.id()),
                        meta.pos_z(vox.id()),
                        vox.value()] for vox in voxels.as_vector()
                      ]
                     ) 
    
    cluster_sum = np.sum([p.energy.sum() for p in label.Particles()])
    input_sum  = np.sum([np.sum([edep.e for edep in p.pcloud]) for p in data])
    input_unass= np.sum([edep.e for edep in data.unassociated_edeps])
    energy_sum = label._energies.sum()
    energy_num = label._energies.size()
    pcloud_sum = np.sum(pcloud[:,3])
    pcloud_num = pcloud.shape[0]
    voxels_sum = np.sum(voxels[:,3])
    voxels_num = voxels.shape[0]
    unass_sum  = np.sum([vox.value() for vox in label._unassociated_voxels.as_vector()])
    
    if verbose:
        print('  Raw image    :',voxels_num,'voxels with the sum',voxels_sum)
        print('  Packets      :',pcloud_num,'packets with the sum',pcloud_sum)
        print('  Input cluster:',input_sum)
        print('  Input unass. :',input_unass)
        print('  Label image  :',energy_num,'voxels with the sum',energy_sum)
        print('  Label cluster:',cluster_sum)
        print('  Unassociated :',unass_sum)
        print('  Label image - (Cluster sum + Unassociated)',energy_sum - (cluster_sum + unass_sum))
        print('  Label image - Raw image',energy_sum - voxels_sum)
        print('  Packets - Raw image',pcloud_sum - voxels_sum)

    log['raw_image_sum'].append(voxels_sum)
    log['raw_image_npx'].append(voxels_num)
    log['raw_packet_sum'].append(pcloud_sum)
    log['raw_packet_num'].append(pcloud_num)
    log['in_cluster_sum'].append(input_sum)
    log['in_unass_sum'].append(input_unass)
    log['out_image_sum'].append(energy_sum)
    log['out_image_num'].append(energy_num)
    log['out_cluster_sum'].append(cluster_sum)
    log['out_unass_sum'].append(unass_sum)
    
# Fill SuperaAtomic class and hand off to label-making
def run_supera(out_file='larcv.root',
               in_file='',
               config_key='',
               num_events=-1,
               num_skip=0,
            #    ignore_bad_association=True,
               save_log=None,
               verbose=False):

    start_time = time.time()

    writer = get_iomanager(out_file)
  
    driver = get_flow2supera(config_key)

    reader = flow2supera.reader.InputReader(driver.parser_run_config(), in_file,config_key)

    id_vv = ROOT.std.vector("std::vector<unsigned long>")()
    value_vv = ROOT.std.vector("std::vector<float>")()

    id_v=ROOT.std.vector("unsigned long")()
    value_v=ROOT.std.vector("float")()

    if num_events < 0:
        num_events = len(reader)

    print("[run_supera] startup {:.3e} seconds".format(time.time() - start_time))

    LOG_KEYS  = ['event_id','time_read','time_convert','time_generate', 'time_store', 'time_event']
    LOG_KEYS += ['raw_image_sum','raw_image_npx','raw_packet_sum','raw_packet_num',
    'in_cluster_sum','in_unass_sum','out_image_sum','out_image_num',
    'out_cluster_sum','out_unass_sum']

    logger = dict()
    if save_log:
        for key in LOG_KEYS:
            logger[key]=[]
        driver.log(logger)
    
    print(f"[run_supera] Processing {len(reader)} entries ")
    for entry in range(len(reader)):

        if num_skip and entry < num_skip:
            continue

        if num_events <= 0:
            break

        num_events -= 1 

        print(f'\n----- [run_supera] Processing Entry {entry} -----\n')

        t0 = time.time()
        input_data = reader.GetEntry(entry)
        #reader.EventDump(input_data)
        time_read = time.time() - t0
        print("[run_supera] reading input   {:.3e} seconds".format(time_read))

        t1 = time.time()
        EventInput = driver.ReadEvent(input_data)
        time_convert = time.time() - t1
        print("[run_supera] data conversion {:.3e} seconds".format(time_convert))
      
        t2 = time.time()
        driver.GenerateImageMeta(EventInput)

        # Perform an integrity check
        if save_log:
            log_supera_integrity_check(EventInput, driver, logger, verbose=False)
            
        meta   = larcv_meta(driver.Meta())
        tensor_hits = writer.get_data("sparse3d", "hits")
        
        if not reader._is_sim:
            driver.Meta().edep2voxelset(EventInput.unassociated_edeps).fill_std_vectors(id_v, value_v)
            larcv.as_event_sparse3d(tensor_hits, meta, id_v, value_v)
      
        if reader._is_sim:            
            if input_data.trajectories is None:
                print(f'[run_supera] WARNING skipping this entry {entry} as it appears to be "empty" (no truth association found, non-unique event id, etc.)')
                continue
            driver.Meta().edep2voxelset(driver._edeps_all).fill_std_vectors(id_v, value_v)
            larcv.as_event_sparse3d(tensor_hits, meta, id_v, value_v)
            driver.GenerateLabel(EventInput) 
            time_generate = time.time() - t2
            print("[run_supera] label creation  {:.3e} seconds".format(time_generate))

            # Start data store process
            t3 = time.time()

            result = driver.Label()
            tensor_energy = writer.get_data("sparse3d", "pcluster")
            result.FillTensorEnergy(id_v, value_v)
            larcv.as_event_sparse3d(tensor_energy, meta, id_v, value_v)
            
            # Check the input image and label image match in the voxel set
            ids_input = np.array([v.id() for v in tensor_energy.as_vector()])
            ids_label = np.array([v.id() for v in tensor_hits.as_vector()])
 
            assert np.allclose(ids_input,ids_label), '[run_supera] ERROR: the label and input data has different set of voxels'

            tensor_semantic = writer.get_data("sparse3d", "pcluster_semantics")
            result.FillTensorSemantic(id_v, value_v)
            larcv.as_event_sparse3d(tensor_semantic,meta, id_v, value_v)

            cluster_energy = writer.get_data("cluster3d", "pcluster")
            result.FillClustersEnergy(id_vv, value_vv)
            larcv.as_event_cluster3d(cluster_energy, meta, id_vv, value_vv)

            cluster_dedx = writer.get_data("cluster3d", "pcluster_dedx")
            result.FillClustersdEdX(id_vv, value_vv)
            larcv.as_event_cluster3d(cluster_dedx, meta, id_vv, value_vv)

            particle = writer.get_data("particle", "pcluster")
            for p in result._particles:
                if not p.valid:
                    continue
                larp = larcv_particle(p)
                particle.append(larp)
            
            #Fill mc truth neutrino interactions
            interaction = writer.get_data("neutrino", "mc_truth")
            for ixn in input_data.interactions:
                if isinstance(ixn,np.void):
                    continue
                larn = larcv_neutrino(ixn)
                interaction.append(larn)
            
            time_store = time.time() - t3
            print("[run_supera] storing output  {:.3e} seconds".format(time_store))

        #propagating trigger info
        trigger = writer.get_data("trigger", "base")
        trigger.id(int(input_data.event_id))  # fixme: this will need to be different for real data?
        trigger.time_s(int(input_data.t0))
        trigger.time_ns(int(1e9 * (input_data.t0 - trigger.time_s())))   

        # TODO fill the run ID 
        writer.set_id(0, 0, int(input_data.event_id))
        writer.save_entry()

        time_event = time.time() - t0
        print("[run_supera] driver total    {:.3e} seconds".format(time_event))

    writer.finalize()

    
    if save_log:
        np.savez('log_flow2supera.npz',**logger)

    end_time = time.time()
    
    print("\n----- [run_suera] finished -----\n")
    print("[run_supera] Total processing time in s: ", end_time-start_time)








