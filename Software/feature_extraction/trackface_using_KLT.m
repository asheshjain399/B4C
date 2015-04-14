function status_tracker = trackface_using_KLT(params_in)
    
    
    % params_in
    %   frame_start: Starting frame number
    %   frame_end: End frame number
    %   videoPath: Path to the directory containing video
    %   videoFilename: video filename without extension
   
    show_video = true;
    
    THRESH_MOVEMENT = 2;
    scale = 0.5;
    videoPath = params_in.videoPath;
    videoFilename = params_in.videoFilename;
    framesPath = [videoPath '/frames_' videoFilename];
    
    faceDetector = vision.CascadeObjectDetector();
    profileDetector = vision.CascadeObjectDetector('ClassificationModel','ProfileFace');
    pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
    
    % Ignoring first few frames
    if isfield(params_in, 'frame_start')
        frame_start = params_in.frame_start;
    else
        frame_start = 1;
    end;
    
    % Ignoring first few frames
    if isfield(params_in, 'frame_end')
        frame_end = params_in.frame_end;
    else
        frame_end = 1;
    end;
    
    % Reading starting frame
    videoFrame = imread([framesPath '/' num2str(frame_start) '.jpg']);
    videoFrame = imresize(videoFrame,scale);
    
    height = size(videoFrame,1);
    width = size(videoFrame,2);
    mask = zeros(height,width);
    
    if strcmp(videoFilename(1:8) , '20141025')
        h1 = floor(1*height/20.0);
        h2 = floor(7*height/10.0);
        w1 = floor(1*width/6.0);
        w2 = floor(7*width/10.0);
        mask(h1:h2,w1:w2)=1.0;
        mask_rect = [w1,h1,w2-w1,h2-h1];    
    elseif strcmp(videoFilename(1:8) , '20141126')
        h1 = floor(1*height/11.0);
        h2 = floor(7.3*height/10.0);
        w1 = floor(2.8*width/10.0);
        w2 = floor(8*width/10.0);
        mask(h1:h2,w1:w2)=1.0;
        mask_rect = [w1,h1,w2-w1,h2-h1];    
    elseif strcmp(videoFilename(1:8) , '20141123')
        h1 = floor(1*height/11.0);
        h2 = floor(7.3*height/10.0);
        w1 = floor(3.3*width/10.0);
        w2 = floor(8.2*width/10.0);
        mask(h1:h2,w1:w2)=1.0;
        mask_rect = [w1,h1,w2-w1,h2-h1];    
    elseif strcmp(videoFilename(1:8) , '20141101') || strcmp(videoFilename(1:8) , '20141102')  || strcmp(videoFilename(1:8) , '20141105')  || strcmp(videoFilename(1:8) , '20141107')  || strcmp(videoFilename(1:8) , '20141108')
        h1 = max(1,floor(0.01*height/11.0));
        h2 = floor(6.4*height/10.0);
        w1 = floor(2.6*width/10.0);
        w2 = floor(7.2*width/10.0);
        mask(h1:h2,w1:w2)=1.0;
        mask_rect = [w1,h1,w2-w1,h2-h1];    
    elseif strcmp(videoFilename(1:8) , '20141220')
        h1 = floor(0.9*height/10.0);
        h2 = floor(8.3*height/10.0);
        w1 = floor(2.3*width/10.0);
        w2 = floor(7.8*width/10.0);
        mask(h1:h2,w1:w2)=1.0;
        mask_rect = [w1,h1,w2-w1,h2-h1];    
    else
        mask(floor(1*height/10.0):floor(8*height/10.0),floor(2*width/6.0):floor(4*width/6.0))=1.0;
        mask_rect = [floor(2*width/6.0),floor(1*height/10.0),floor(2*width/6.0),floor(7*height/10.0)];
    end;
    
    
   
    status_tracker.completed_successfully = false;    
    if exist([framesPath '/' num2str(frame_start) '_bbox.txt'],'file') == 2
        f = fopen([framesPath '/' num2str(frame_start) '_bbox.txt'],'r');
        bbox = fscanf(f,'%f %f %f %f')';
        fclose(f);
    else
        bbox = step(faceDetector, videoFrame);

        if numel(bbox) == 0
            status_tracker.initial_box_found = false;
            return
        end;

        [~,idx]=max(bbox(:,3));
        bbox = bbox(idx,:);

        if rectint(bbox,mask_rect)/rectint(bbox,bbox) < 0.6
            status_tracker.initial_box_outside_mask = true;
            return
        end;
    end;
    status_tracker.initial_box_found = true;
    status_tracker.initial_box_outside_mask = false;

    addpath ../sift/
    addpath ../Utils/
    addpath ../.

    if show_video
        videoPlayer  = vision.VideoPlayer('Position',[300 300 width+30 height+30]);
    end;

    videoFrame_old = videoFrame;
    bbox_old = bbox;
    frames2_match = [];
    frame_number = frame_start;
    frames_match = [];
    
    vidobj = VideoWriter([framesPath '/' num2str(frame_start) '_' num2str(frame_end) '_onlypoints_KLT.avi']);
    open(vidobj);
    
    points = detectMinEigenFeatures(rgb2gray(videoFrame_old), 'ROI', round(bbox_old));
    points = points.Location;
    initialize(pointTracker, points, videoFrame_old);
    
    bboxPoints = bbox2points( bbox_old );
    bbox = bbox_old;
    writeMatchPoints([],frame_number,framesPath,true);
    xpoints = [];
    tpoints = [];
    while frame_number <= frame_end
        oldPoints = points;
        writeKLTPoints(points,frame_number,framesPath);
        xpoints = [xpoints ;points(:,1)];
        tpoints = [tpoints;repmat(frame_number-frame_start+1,size(points,1),1)];
        
        if exist([framesPath '/' num2str(frame_number+1) '.jpg'],'file') == 2
            videoFrame = imread([framesPath '/' num2str(frame_number+1) '.jpg']);
            videoFrame = imresize(videoFrame,scale);
        else
            close(vidobj);
            return
        end;
        
        [points, isFound] = step(pointTracker, (videoFrame));
        visiblePoints = points(isFound, :);
        oldInliers = oldPoints(isFound, :);
        
        if size(visiblePoints,1) >= 4
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'projective', 'MaxDistance', 4);
            bboxPoints = transformPointsForward(xform, bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);  
            writeKLTTransform(xform,frame_number,framesPath);
        end;
        
        points_inside_mask = FilteringOutsideMaskPoints(visiblePoints',visiblePoints',mask);
        visiblePoints(find(points_inside_mask==0),:)=[];
        oldInliers(find(points_inside_mask==0),:)=[];
        
        writeBBOX(bbox,frame_number,framesPath);
        writeMatchPoints(oldInliers',frame_number,framesPath,false);
        writeMatchPoints(visiblePoints',frame_number+1,framesPath,true);
        
        %videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face');
        
        if size(visiblePoints,1) > 0
            temp_holder=zeros(3,size(visiblePoints,1));
            temp_holder(1:2,:)=visiblePoints';
            temp_holder(3,:)=004;
            videoOut = insertShape(videoFrame,'FilledCircle',temp_holder','Color','green');  
        end
        
        if show_video
            step(videoPlayer, videoOut);
        end;
        writeVideo(vidobj,videoOut);
     
        if size(visiblePoints,1) > 0
            bbox(1:2) = bbox(1:2) + (mean(visiblePoints - oldInliers,1));
        end;
        
        if mod(frame_number,5) == 0
            fname = [framesPath '/' num2str(frame_number+1) '_bbox.txt'];
            if false %exist(fname,'file') == 0
                fptr = fopen(fname);
                bbox_saved = fscanf(fptr,'%f',[1 inf]);
                f = 0.2;
                bbox = bbox + f*(bbox_saved - bbox);
                fclose(fptr);
            else    
                bbox_front = step(faceDetector, videoFrame);
                bbox_profile = step(profileDetector, videoFrame);

                front_intersection = 0.0;
                if numel(bbox_front)>0
                    front_intersection = zeros(1,size(bbox_front,1));
                    for i = 1:1:size(bbox_front,1)
                        front_intersection(i) = rectint(bbox_front(i,:),bbox)/rectint(bbox,bbox);
                    end
                    [front_intersection,idx] = max(front_intersection);
                    bbox_front = bbox_front(idx,:);
                end

                profile_intersection = 0.0;
                if numel(bbox_profile)>0
                    profile_intersection = zeros(1,size(bbox_profile,1));
                    for i = 1:1:size(bbox_profile,1)
                        profile_intersection(i) = rectint(bbox_profile(i,:),bbox)/rectint(bbox,bbox);
                    end
                    [profile_intersection,idx] = max(profile_intersection);
                    bbox_profile = bbox_profile(idx,:);
                end

                f = 0.5;
                if (front_intersection > profile_intersection) && front_intersection > 0.4
                    bbox = bbox + f*(bbox_front - bbox);
                end
                if (profile_intersection > front_intersection) && profile_intersection > 0.4
                    bbox = bbox + f*(bbox_profile - bbox);
                end
            end;
        end
        
        bbox = max([1,1,1,1],bbox);
        in = checkbbox(bbox,visiblePoints');
        visiblePoints(~in,:)=[];
        if size(visiblePoints,1) < 20
            visiblePoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', round(bbox));
            visiblePoints = visiblePoints.Location;
        end;
        
        
        points = visiblePoints;
        if size(points,1) == 0
            disp('\nReturning incomplete -- no GoodFeaturesToTrack\n');
            close(vidobj);
            return
        end;
        setPoints(pointTracker, points);
        frame_number = frame_number + 1;

    end;
    close(vidobj);
    h = figure;
    plot(tpoints,xpoints,'.');
    print(h,'-dpsc',[framesPath '/' num2str(frame_start) '_' num2str(frame_end) '_KLT.ps']);
    close all;
    status_tracker.completed_successfully = true;  
end


function writeBBOX(bbox,frame,path)
    fname = [path '/' num2str(frame) '_bbox_KLT.txt'];
    f = fopen(fname,'w');
    fprintf(f,[num2str(bbox(1)) ' ' num2str(bbox(2)) ' ' num2str(bbox(3)) ' ' num2str(bbox(4))]);
    fclose(f);

end

function writeMatchPoints(match,frame,path,prev)
    if prev
        fname = [path '/' num2str(frame) '_match_prev_KLT.txt'];
    else
        fname = [path '/' num2str(frame) '_match_next_KLT.txt'];
    end;
    dlmwrite(fname,match');
   
end

function writeKLTPoints(match,frame,path)
    fname = [path '/' num2str(frame) '_KLTPoints.txt'];
    dlmwrite(fname,match);
   
end

function writeKLTTransform(match,frame,path)
    fname = [path '/' num2str(frame) '_'  num2str(frame+1) '_KLTTransform.txt'];
    dlmwrite(fname,match.T);
   
end
