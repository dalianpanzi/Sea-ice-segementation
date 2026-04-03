import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
np.random.seed(42)
import geopandas as gpd
from eolearn.core import EOTask, EOPatch, EOWorkflow, FeatureType, OverwritePermission, SaveTask, EOExecutor
from eolearn.core import EONode, OutputTask, linearly_connect_tasks
from eolearn.io import SentinelHubInputTask
from sentinelhub import DataCollection
from sentinelhub import BBoxSplitter
from sentinelhub import SHConfig
# from eolearn.mask import  AddMultiCloudMaskTask, AddValidDataMaskTask
from eolearn.geometry import VectorToRasterTask, ErosionTask
from pyproj import Transformer
from sentinelhub import CRS, BBox, DataCollection
from eolearn.features import SimpleFilterTask
import os
from PIL import Image

class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

class CountValid(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """
    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name
    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name, np.count_nonzero(eopatch.mask[self.what], axis=0))
        return eopatch

# CUSTOM EOTASK WHICH ADDS A SCALAR FEATURE WITH % OF IMAGE THAT IS VALID (AVAILABLE AND NOT CLOUDY)
class AddValidDataMaskTask(EOTask):
    def execute(self, eopatch):
        eopatch.mask["VALID_DATA"] = eopatch.mask["IS_DATA"].astype(bool) & ~(eopatch.mask["CLM"].astype(bool))
        return eopatch

class AddValidDataCoverage(EOTask):
    def execute(self, eopatch):
        valid_data = eopatch.mask["VALID_DATA"]
        # valid_data = eopatch.get_feature(FeatureType.MASK, 'VALID_DATA')
        time, height, width, channels = valid_data.shape
        coverage = np.apply_along_axis(calculate_coverage, 1,
                                       valid_data.reshape((time, height * width * channels)))
        eopatch.scalar["COVERAGE"] = coverage[:, np.newaxis]
        # eopatch.add_feature(FeatureType.SCALAR, 'COVERAGE', coverage[:, np.newaxis])
        return eopatch

class ValidDatePredicate:
    def __init__(self, days):
        self.days = days
    def __call__(self, array):
        return array.weekday() in self.days
class ValidDataCoveragePredicate:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold


class TimeRaster(EOTask):

    def __init__(self, chart_dir, base_len,id,ic_gdf):
        self.chart_dir = chart_dir
        self.base_len = base_len
        self.id=id
        self.ic_gdf=ic_gdf

    def execute(self, eopatch):

        chart_folders = [name for name in os.listdir(chart_dir)]  # all the sea ice chart folders available
        chart_dates = np.array([datetime.datetime.strptime(name[6:], '%Y%m%d') for name in
                                os.listdir(chart_dir)])  # all the dates available
        chart_dates = chart_dates + datetime.timedelta(
            hours=12)  # add 12 hours to the date so it is in the middle of the day

        # get the ice chart dates corresponding to the image dates
        sat_dates = np.array(eopatch.timestamps)  # get the dates for each image in the eopatch
        mask_dates = []  # this will hold the dates of the ice chart associated with each satellite image
        for date in sat_dates:
            closest_date_id = np.argsort(abs(date - chart_dates))[0]
            closest_date = chart_dates[closest_date_id]  # closest ice chart date to the image date
            mask_dates.append(closest_date)

        # get the file paths of the ice charts corresponding to the images
        chart_paths = []
        for date in mask_dates:
            names = [name for name in chart_folders if
                     date.strftime('%Y%m%d') in name]  # get all the the folder names that match the date
            folder = max(names,
                         key=len)  # get the longest named folder (this is the most recent revision in case the folder was updated)
            if len(folder) > self.base_len:  # if the folder has been updated
                file = folder[:self.base_len]  # then the file will only contain the first base_len characters
            else:
                file = folder  # otherwise the filename is the same as the folder
            chart_paths.append(
                self.chart_dir + '/' + folder + '/' + file + '.shp')  # file path of the ice chart shapefile

        # gather all of the ice charts into a single time-dependent feature
        times, length, width, channels = eopatch.data['BANDS'].shape
        timed_mask = np.zeros(shape=(times, length, width, 1), dtype=np.int16)
        geometry_id=self.ic_gdf.loc[self.id,'geometry']

        for count, path in enumerate(chart_paths):  # for each ice chart path
            chart = gpd.read_file(path)  # get the ice chart
            chart.fillna(100, inplace=True)  # nans are land, we will fill them with 255
            chart['CT'] = chart['CT'].astype('int16')
            chart = chart.to_crs(region.crs)  # make sure the chart is in the correct crs

            chart_id=chart[chart.geometry.intersects(geometry_id)]
            #ct_value=chart_id['CT'].values

            # convert the chart to a raster image and store in the eopatch
            add_raster = VectorToRasterTask(chart_id, (FeatureType.MASK_TIMELESS, 'TEMP'),
                                            values_column='CT', raster_shape=(FeatureType.DATA, 'BANDS'),
                                            raster_dtype=np.int16,
                                            no_data_value=100)  # no data will retult in the mask assuming land
            add_raster(eopatch)
            timed_mask[count] = eopatch.mask_timeless['TEMP']  # append the resulting raster image to our numpy array

        eopatch.mask['ICE_CHART'] = timed_mask  # store the numpy array in the eopatch mask feature
        return eopatch
      
if __name__ == '__main__':
        target_date='0322'
        ice_gdf=gpd.read_file('.\data\seepicechaer\ARCTIC2024{}\ARCTIC2024{}.shp'.format(target_date,target_date))# from nsidc
        ice_gdf = ice_gdf.to_crs('EPSG:32659')  # arctic168-174   赤道-84  32659   'EPSG:6931'EPSG:32659
        region = ice_gdf.dropna()  # 去除包含无数据的多边形（对应于陆地）
        region = region.geometry.unary_union  # 将所有水域区域的几何形状取并集，得到一个包含所有水域的整体几何形状
        region = gpd.GeoDataFrame(geometry=[region], crs=ice_gdf.crs)
        region_shape = region.geometry.values[-1]
        bbox_splitter = BBoxSplitter([region_shape], region.crs, (70, 70))
        bbox_list = np.array(bbox_splitter.get_bbox_list())
        info_list = np.array(bbox_splitter.get_info_list())
        for n, info in enumerate(info_list):
            info['index'] = n

        geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list]
        idxs = [info['index'] for info in info_list]
        idxs_x = [info['index_x'] for info in info_list]
        idxs_y = [info['index_y'] for info in info_list]

        gdf = gpd.GeoDataFrame({'index': idxs, 'index_x': idxs_x, 'index_y': idxs_y},
                               crs=region.crs,
                               geometry=geometry)
        ID = 1045
        '''    仅需第一次看图
        patchIDs = []
        for idx, [bbox, info] in enumerate(zip(bbox_list, info_list)):
            if ID < len(info_list) and (abs(info['index_x'] - info_list[ID]['index_x']) <= 1 and
                                        abs(info['index_y'] - info_list[ID]['index_y']) <= 1):
                patchIDs.append(idx)
        patchIDs = np.transpose(np.fliplr(np.array(patchIDs).reshape(3, 3))).ravel()

        fig, ax = plt.subplots(figsize=(20, 20))
        gdf.plot(ax=ax, facecolor='w', edgecolor='r', alpha=0.5)
        region.plot(ax=ax, facecolor='w', edgecolor='b', alpha=0.5)
        ax.set_title('Arctic', fontsize=25)
        for bbox, info in zip(bbox_list, info_list):
            geo = bbox.geometry
            ax.text(geo.centroid.x, geo.centroid.y, info['index'], ha='center', va='center', size=12)

        gdf[gdf.index.isin(patchIDs)].plot(ax=ax, facecolor='g', edgecolor='r', alpha=0.5)

        plt.axis('off')
        plt.show()
        '''
        band_names = ['B03', 'B04', 'B08']  # false color bands
        config = SHConfig()
        config.sh_client_id = '**********'  #一个月过期
        config.sh_client_secret = '**************'
        add_data = SentinelHubInputTask(
            bands_feature=(FeatureType.DATA, 'BANDS'),  # location where the images will be stored in the EOPatch
            bands=band_names,  # bands to collect in the image
            resolution=(200, 200),  # resolution of the images in m
            maxcc=0.8,  # maximum cloud cover to allow 1=100%
            time_difference=datetime.timedelta(minutes=120),
            # if two images are this close to each other they are considered the same
            data_collection=DataCollection.SENTINEL2_L1C,
            additional_data=[(FeatureType.MASK, 'dataMask', 'IS_DATA'),
                             # also download the 'is_data' and cloud cover masks from sentinelhub
                             (FeatureType.MASK, 'CLM'),
                             ],
            config=config
        )

        # add_valid_mask = AddValidDataMaskTask(predicate=calculate_valid_data_mask, valid_data_feature='VALID_DATA')
        add_valid_mask = AddValidDataMaskTask()
        add_coverage = AddValidDataCoverage()

        cloud_coverage_threshold = 0.05
        remove_cloudy_scenes = SimpleFilterTask((FeatureType.MASK, 'VALID_DATA'),
                                                ValidDataCoveragePredicate(cloud_coverage_threshold))
        path_out='./data/{}eopatch/'.format(ID)

        if not os.path.isdir(path_out):
            os.makedirs(path_out)

        save = SaveTask(path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

        chart_dir = "./data/seepicechaer"
        base_len = 31

        time_raster = TimeRaster(chart_dir, base_len,ID,gdf)

        workflow_nodes = linearly_connect_tasks(
            add_data,
            add_valid_mask,
            add_coverage,
            # remove_cloudy_scenes,
            time_raster,
            save,
        )

        input_node = workflow_nodes[0]
        save_node = workflow_nodes[-1]

        workflow = EOWorkflow(workflow_nodes)

        # workflow.dependency_graph()

        time_interval = ['2024-3-17', '2024-5-29']
        execution_args = []

        for idx, bbox in enumerate(bbox_list[[ID]]):
            execution_args.append(
                {
                    input_node: {'bbox': bbox, 'time_interval': time_interval},
                    save_node: {'eopatch_folder': 'eopatch_{}'.format(target_date)}
                }
            )

        executor = EOExecutor(workflow, execution_args, save_logs=True)
        executor.run()

