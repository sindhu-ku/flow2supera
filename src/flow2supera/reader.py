import h5py 
import h5flow
import numpy as np
from ROOT import supera
from yaml import Loader
import yaml
import os
import flow2supera


class InputEvent:
    event_id = -1
    true_event_id = -1
    segments = None
    hit_indices = None
    hits = None
    backtracked_hits = None
    calib_final_hits  = None
    trajectories = None
    interactions = []
    t0 = -1
    segment_index_min = -1
    event_separator = ''
    flashes = []
    light_events = None


class FlowReader:
    
    def __init__(self, parser_run_config, input_files=None,config=None):
        self._input_files = input_files
        if not isinstance(input_files, str):
            raise TypeError('Input file must be a str type')
        self._event_ids = None
        self._event_t0s = None
        self._flashes = None
        self._event_hit_indices = None
        self._hits = None
        self._backtracked_hits = None
        self._segments = None
        self._trajectories = None
        self._interactions = None
        self._run_config = parser_run_config
        self._is_sim = False
        self._is_mpvmpr= False
        if config:
            if os.path.isfile(config):
                file=config
            else:
                file=flow2supera.config.get_config(config)
            with open(file,'r') as f:
                cfg=yaml.load(f.read(),Loader=Loader)
                if 'Type' in cfg.keys():
                    self._is_sim=cfg.get('Type')[0]=='sim'
                    self._is_mpvmpr=cfg.get('Type')[1]=='mpvmpr'  
                

        
        if input_files:
            self.ReadFile(input_files)

    def __len__(self):
        if self._event_ids is None: return 0
        return len(self._event_ids)

    
    def __iter__(self):
        for entry in range(len(self)):
            yield self.GetEvent(entry)

    def ReadFile(self, input_files, verbose=False):
        
        print('Reading input file...')

        # H5Flow's H5FlowDataManager class associated datasets through references
        # These paths help us get the correct associations
        events_path = 'charge/events/data'
        event_hit_indices_path = 'charge/events/ref/charge/calib_prompt_hits/ref_region/'
        calib_prompt_hits_path = 'charge/calib_prompt_hits/data' 
    
        backtracked_hits_path = 'mc_truth/calib_prompt_hit_backtrack/data'
        interactions_path = 'mc_truth/interactions/data'
        segments_path = 'mc_truth/segments/data'
        trajectories_path = 'mc_truth/trajectories/data'
        
        light_events_path = 'light/events/data'
        flash_path = 'light/flash/data'
        flash_light_ref_path = 'light/events/ref/light/flash/ref_region'
        charge_light_ref_path = 'charge/events/ref/light/events/ref_region'


        #for f in input_files:
        flow_manager = h5flow.data.H5FlowDataManager(input_files, 'r')
        with h5py.File(input_files, 'r') as fin:
            events = flow_manager[events_path]
            self._event_ids = events['id']
            self._event_t0s = events['unix_ts'] + events['ts_start']/1e7 #ts_start is in ticks and 0.1 microseconds per tick for charge readout
            self._event_hit_indices = flow_manager[event_hit_indices_path]
            self._hits = flow_manager[calib_prompt_hits_path]
            self._light_event_indices = flow_manager[charge_light_ref_path]
            self._light_events = flow_manager[light_events_path]
            self._flash_indices =  flow_manager[flash_light_ref_path]
            self._flashes = flow_manager[flash_path]
            if self._is_sim:
                self._segments = flow_manager[segments_path]
                self._trajectories = flow_manager[trajectories_path]
                self._interactions = flow_manager[interactions_path]
                self._backtracked_hits = flow_manager[backtracked_hits_path]
    
    def GetNeutrinoIxn(self, ixn, ixn_idx):
        
        supera_neutrino = supera.Neutrino()
        
        supera_neutrino.id = int(ixn_idx)
        supera_neutrino.interaction_id = int(ixn['vertex_id']) 
        supera_neutrino.target = int(ixn['target'])
        supera_neutrino.vtx = supera.Vertex(ixn['x_vert'], ixn['y_vert'], ixn['z_vert'], ixn['t_vert'])
        supera_neutrino.pdg_code = int(ixn['nu_pdg'])
        supera_neutrino.lepton_pdg_code = int(ixn['lep_pdg'])  
        supera_neutrino.energy_init = ixn['Enu']
        supera_neutrino.theta = ixn['lep_ang']
        supera_neutrino.momentum_transfer =  ixn['Q2']
        supera_neutrino.momentum_transfer_mag =  ixn['q3']
        supera_neutrino.energy_transfer =  ixn['q0']
        supera_neutrino.bjorken_x = ixn['x']
        supera_neutrino.inelasticity = ixn['y']
        supera_neutrino.px = ixn['nu_4mom'][0]
        supera_neutrino.py = ixn['nu_4mom'][1]       
        supera_neutrino.pz = ixn['nu_4mom'][2]
        supera_neutrino.lepton_p = ixn['lep_mom']
        if(ixn['isCC']): supera_neutrino.current_type = 0
        else: supera_neutrino.current_type = 1
        supera_neutrino.interaction_mode = int(ixn['reaction'])
        supera_neutrino.interaction_type = int(ixn['reaction'])   
        
        return supera_neutrino  
    
    def GetFlash(self, flash, t0):

        supera_flash = supera.Flash()
        supera_flash.id = int(flash['id'])
        supera_flash.time = flash['hit_time_range'][0]*1e-9 + t0
        supera_flash.timeWidth = (flash['hit_time_range'][1] - flash['hit_time_range'][0])*1e-9
        supera_flash.PEPerOpDet = np.array(flash['deconv_sum']).flatten() #*0.022857 #adc charge to pe conversion
        supera_flash.tpc = int(flash['tpc'])

        return supera_flash
    
    # To truth associations go as hits -> segments -> trajectories
    def GetEventIDFromSegments(self, backtracked_hits, segments):

        if len(backtracked_hits) == 0: 
            print('Returning empty event')
            return -1
    
        seg_ids = np.unique(np.concatenate([bhit['segment_ids'][bhit['segment_ids']!=-1] for bhit in backtracked_hits]))

        st_event_id = np.unique(segments[seg_ids]['event_id'])
    
        assert len(st_event_id)==1, f'Found >1 unique "event_id" from backtracked segments ({st_event_id})'


        st_event_id = st_event_id[0]
        
        return st_event_id
    
    
    def GetEvent(self, event_index):
        
        if event_index >= len(self._event_ids):
            print('Entry {} is above allowed entry index ({})'.format(event_index, len(self._event_ids)))
            print('Invalid read request (returning None)')
            return None
        
        result = InputEvent()

        result.event_id = self._event_ids[event_index]


        result.t0 = self._event_t0s[result.event_id] 
         
        #Find hits associated with the event
        result.hit_indices = self._event_hit_indices[result.event_id]
        hit_start_index = self._event_hit_indices[result.event_id][0]
        hit_stop_index  = self._event_hit_indices[result.event_id][1]
        result.hits = self._hits[hit_start_index:hit_stop_index]
    
        #Light association
        event_flashes = []

        #link the light events associated with the charge event
        light_events_start = self._light_event_indices[result.event_id][0]
        light_events_stop = self._light_event_indices[result.event_id][1]

        result.light_events = self._light_events[light_events_start:light_events_stop]

        #link the flashes associated with the light events
        for lev in result.light_events:
            flash_start = self._flash_indices[lev['id']][0]
            flash_end = self._flash_indices[lev['id']][1]
            event_flashes.extend(self._flashes[flash_start:flash_end])

        result.flashes = []

        for flash in event_flashes:
            supera_flash = self.GetFlash(flash, result.t0) #fix this with the actual light trigger time when the variable is added to flash
            result.flashes.append(supera_flash)
        
        if not self._is_sim:
            return result
       
        #Find trajectories and segments using the backtracked hits and true event id
        result.backtracked_hits = self._backtracked_hits[hit_start_index:hit_stop_index]
    
        st_event_id = self.GetEventIDFromSegments(result.backtracked_hits,self._segments)
        
        result.segments = self._segments[self._segments['event_id']==st_event_id]
        result.trajectories = self._trajectories[self._trajectories['event_id']==st_event_id]
        
        if self._is_mpvmpr or st_event_id == -1:
            return result
        
        #Find true neutrino interactions associated with the reco events
        result.interactions = []
        
        result.true_event_id = st_event_id      
        interactions_array  = np.array(self._interactions)
        event_interactions = interactions_array[interactions_array['event_id'] == result.true_event_id]
        for ixn_idx, ixn in enumerate(event_interactions):
            supera_nu = self.GetNeutrinoIxn(ixn, ixn_idx)
            result.interactions.append(supera_nu)  
            
        return result  
 


    def EventDump(self, input_event):
        print('-----------EVENT DUMP-----------------')
        print('Event ID {}'.format(input_event.event_id))
        print('Event t0 {}'.format(input_event.t0))
        print('Event hit indices (start, stop):', input_event.hit_indices)
        print('Hits shape:', input_event.hits.shape)
        print('Associated light events:', len(input_event.light_events))
        print('Associated flashes:', len(input_event.flashes))
        if self._is_sim and len(input_event.hits) !=0:
            print('True event ID {}'.format(input_event.true_event_id))
            print('Backtracked hits len:', len(input_event.backtracked_hits))
            print('Segments in this event:', len(input_event.segments))
            print('Trajectories in this event:', len(input_event.trajectories))
            print('Interactions in this event:', len(input_event.interactions))



