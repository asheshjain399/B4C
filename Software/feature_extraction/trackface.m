function [stin, status_tracker, params_video] = trackface(params_in)
    
    
    THRESH_MOVEMENT = 2;
    scale = 0.5;
    videoPath = params_in.videoPath;
    videoFilename = [params_in.videoFilename 'A.mov'];
    paramsPath = params_in.paramsPath;
    
    %videoPath = '/media/DATAPART1/assisitive_driving/raw_data/driver_3/DCIMA/NORMAL/';
    %videoFilename = ['20141018_121403A.mov'];
    faceDetector = vision.CascadeObjectDetector();
    profileDetector = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
    videoFileReader = vision.VideoFileReader([videoPath  '/' videoFilename]);

    % Ignoring first few frames
    if isfield(params_in, 'frame_start')
        frames_ignore = params_in.frame_start;
    else
        frames_ignore = 1;
    end;
    for i = 1:1:frames_ignore
        videoFrame      = step(videoFileReader);
    end;
    videoFrame = imrotate(videoFrame,180);

    videoFrame = imresize(videoFrame,scale);
    height = size(videoFrame,1);
    width = size(videoFrame,2);
    mask = zeros(height,width);
    mask(floor(1*height/10.0):floor(8*height/10.0),floor(2*width/6.0):floor(4*width/6.0))=1.0;
    mask_rect = [floor(2*width/6.0),floor(1*height/10.0),floor(2*width/6.0),floor(7*height/10.0)];
    params_video.scale = scale;
    params_video.THRESH_MOVEMENT = THRESH_MOVEMENT;
    params_video.mask = mask;
    params_video.frame_data = {};
    params_video.videoFilename = videoFilename;
    params_video.videoPath = videoPath;
    
    if isfield(params_in, 'frame_end')
        frame_end = params_in.frame_end;
    else
        frame_end = Inf;
    end;
    
    bbox = step(faceDetector, videoFrame);
    status_tracker.completed_successfully = false;
    
    if numel(bbox) == 0
        status_tracker.initial_box_found = false;
        status_tracker.completed_successfully = false;
        return
    else
        status_tracker.initial_box_found = true;
    end;

    [val,idx]=max(bbox(:,3));
    bbox = bbox(idx,:);

    videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
    figure, imshow(videoOut)
    stin = input('Use or Ignore\n','s');
    close all;
    return
    
    if rectint(bbox,mask_rect)/rectint(bbox,bbox) < 0.6
        status_tracker.initial_box_outside_mask = true;
        status_tracker.completed_successfully = false;
        return
    else
        status_tracker.initial_box_outside_mask = false;
    end;
    

    addpath ../sift/
    addpath ../Utils/
    addpath ../.

    videoInfo    = info(videoFileReader);
    videoPlayer  = vision.VideoPlayer('Position',[300 300 videoInfo.VideoSize+30]);

    bbox_all = [];
    points_all = {};
    videoFrame_old = videoFrame;
    bbox_old = bbox;
    frames2_match = [];
    frame_number = frames_ignore;
    frames_match = [];
    while ~isDone(videoFileReader) && frame_number <= frame_end
        params.box = bbox_old;
        params.frame = frame_number;
        params.match_prev = frames_match;
        params.time = (frame_number-1.0)/30.0;

        videoOut = insertObjectAnnotation(videoFrame_old,'rectangle',bbox_old,'Face');

        if size(frames2_match,1) > 0
            temp_holder=zeros(3,size(frames2_match,2));
            temp_holder(1:2,:)=frames2_match;
            temp_holder(3,:)=002;
            videoOut = insertShape(videoOut,'circle',temp_holder','Color','green');  
            points_all{end+1} = temp_holder;
        else
            points_all{end+1} = [];
        end
        step(videoPlayer, videoOut);

        margin = 0.1;
        bbox_cur_frame = resizeBBOX( bbox_old,margin,size(videoOut,1),size(videoOut,2));

        bbox_prev_frame = bbox_cur_frame;  
        % OR
        %bbox_prev_frame = bbox_old;

        videoFrame = step(videoFileReader);
        videoFrame = imrotate(videoFrame,180);
        videoFrame = imresize(videoFrame,scale);

        [frames1_match,frames2_match] = SiftMatches(videoFrame_old,videoFrame,bbox_prev_frame,bbox_cur_frame);

        % Removing outliers
        t = [];
        if size(frames1_match,2) > 4
            [t,frames1_match_,frames2_match_]=estimateGeometricTransform(frames1_match',frames2_match','projective');
            frames1_match = frames1_match_';
            frames2_match = frames2_match_';
        end

        params.match_next = frames1_match;
        params.transform = t;
        frames_match = frames2_match;
        params_video.frame_data{frame_number} = params;
        %save([paramsPath '/' 'params_' videoFilename '.mat'],'params_video');

        % Filtering points that lie outside the large mask
        points_inside_mask = FilteringOutsideMaskPoints(frames1_match,frames2_match,mask);   
        % Increasing weight for points that lie outside bbox_old but inside
        % bbox w.r.t. original Frame1
        wt = 1.1;
        in = FilteringPointsOutsideInitialTrack( frames1_match, bbox_old, wt );
        % Filtering points that moved less than a threshold distance
        good_movement = FilteringPointsWithSmallMovement( frames1_match, frames2_match, THRESH_MOVEMENT);
        % Points with value 0 are ignored
        points_to_be_ignored = good_movement.*in.*points_inside_mask;
        % Reducing threshold till it find some points
        if max(points_inside_mask) == 1
            tt=0;
            NEW_THRESH_MOVEMENT = THRESH_MOVEMENT;
            while max(points_to_be_ignored) == 0
                tt = tt + 1;
                if tt > 50
                    disp('stop');
                end;
                NEW_THRESH_MOVEMENT = 0.9*NEW_THRESH_MOVEMENT;
                good_movement = FilteringPointsWithSmallMovement( frames1_match, frames2_match, NEW_THRESH_MOVEMENT);
                points_to_be_ignored = good_movement.*in.*points_inside_mask;
            end;
            diff_points = frames2_match - frames1_match;
            diff_points = diff_points.*repmat(points_to_be_ignored,2,1);
            mean_movement = round(mean(diff_points(:,find(points_to_be_ignored>0)),2));
        else
            mean_movement = [0.0;0.0];
        end;


        %{
        margin = 0.1;
        bbox_large = resizeBBOX( bbox_cur_frame,margin,size(videoOut,1),size(videoOut,2));
        videoOut_ = insertObjectAnnotation(videoFrame_old,'rectangle',bbox_cur_frame,'Face');
        videoOut_ = insertObjectAnnotation(videoOut_,'rectangle',bbox_large,'Face');
        %figure,imshow(videoOut_);    
        mean_bg_movement = approxBackgroundMovement( bbox_large,bbox_cur_frame,videoFrame_old,videoFrame,THRESH_MOVEMENT );

        a = 0.1; % ratio of bg to on-the face points
        mean_movement = (mean_movement - a*mean_bg_movement)/(1-a);
        %}
        %figure; clf;
        %title('Click on a keypoint to see its match'); axis off;
        %plotmatches(I1, I2, frames1(1:2,:), frames2(1:2,:), matches, 'Interactive', 2);
        %drawnow;
        close all;

        bbox_old(1) = bbox_old(1) + mean_movement(1);
        bbox_old(2) = bbox_old(2) + mean_movement(2);
        %bbox_old = bbox;
        videoFrame_old = videoFrame;
        frame_number = frame_number + 1;

        if mod(frame_number,5) == 0
            bbox_front = step(faceDetector, videoFrame_old);
            bbox_profile = step(profileDetector, videoFrame_old);

            front_intersection = 0.0;
            if numel(bbox_front)>0
                front_intersection = zeros(1,size(bbox_front,1));
                for i = 1:1:size(bbox_front,1)
                    front_intersection(i) = rectint(bbox_front(i,:),bbox_old)/rectint(bbox_old,bbox_old);
                end
                [front_intersection,idx] = max(front_intersection);
                bbox_front = bbox_front(idx,:);
            end

            profile_intersection = 0.0;
            if numel(bbox_profile)>0
                profile_intersection = zeros(1,size(bbox_profile,1));
                for i = 1:1:size(bbox_profile,1)
                    profile_intersection(i) = rectint(bbox_profile(i,:),bbox_old)/rectint(bbox_old,bbox_old);
                end
                [profile_intersection,idx] = max(profile_intersection);
                bbox_profile = bbox_profile(idx,:);
            end

            f = 0.5;
            if (front_intersection > profile_intersection) && front_intersection > 0.4
                bbox_old = bbox_old + f*(bbox_front - bbox_old);
                %bbox_old(1) = bbox_old(1) + f*(bbox_front(1) - bbox_old(1));
                %bbox_old(2) = bbox_old(2) + f*(bbox_front(2) - bbox_old(2));
            end
            if (profile_intersection > front_intersection) && profile_intersection > 0.4
                bbox_old = bbox_old + f*(bbox_profile - bbox_old);
                %bbox_old(1) = bbox_old(1) + f*(bbox_profile(1) - bbox_old(1));
                %bbox_old(2) = bbox_old(2) + f*(bbox_profile(2) - bbox_old(2));
            end
        end

    end;
    status_tracker.completed_successfully = true;    
end



